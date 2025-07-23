"""
api_encdec_7d.py  —  FastAPI 服务
Updated for vpp_meter.csv with new field names
输入字段: ts, forward_total_active_energy (负荷电量), total_active_power, 
         以及可选的天气和其他字段
输出: 未来7日每15分钟的用电量和功率预测
"""
import os, joblib, holidays, uvicorn, torch, torch.nn as nn
import pandas as pd, numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

ROOT           = "output_pytorch"
PAST_STEPS     = 96 * 5
FUTURE_STEPS   = 96 * 7
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------ 读取列 / Scaler ------------
enc_cols: list = joblib.load(f"{ROOT}/enc_cols.pkl")
dec_cols: list = joblib.load(f"{ROOT}/dec_cols.pkl")
sc_enc  = joblib.load(f"{ROOT}/scaler_enc.pkl")
sc_dec  = joblib.load(f"{ROOT}/scaler_dec.pkl")
sc_y_energy = joblib.load(f"{ROOT}/scaler_y_energy.pkl")
sc_y_power  = joblib.load(f"{ROOT}/scaler_y_power.pkl")

# ------------ LSTM EncDec - 双输出 ------------
class EncDec(nn.Module):
    def __init__(self, d_enc, d_dec, hid=128, drop=.24):
        super().__init__()
        self.enc = nn.LSTM(d_enc, hid, batch_first=True)
        self.dec = nn.LSTM(d_dec, hid, batch_first=True)
        self.dp  = nn.Dropout(drop)
        self.fc_energy = nn.Linear(hid, 1)  # 电量预测
        self.fc_power  = nn.Linear(hid, 1)  # 功率预测
    def forward(self, xe, xd):
        _, (h, c) = self.enc(xe)
        out, _    = self.dec(xd, (h, c))
        out_dp = self.dp(out)
        energy_pred = self.fc_energy(out_dp).squeeze(-1)
        power_pred = self.fc_power(out_dp).squeeze(-1)
        return energy_pred, power_pred

base_model = EncDec(len(enc_cols), len(dec_cols)).to(DEVICE)
base_model.load_state_dict(torch.load(f"{ROOT}/model_weighted.pth", map_location=DEVICE))
base_model.eval()
model_cache = {}

# ------------- Pydantic schema -------------
class Past(BaseModel):
    ts: datetime  # 改为 ts
    forward_total_active_energy: float  # 改为 forward_total_active_energy
    total_active_power: Optional[float] = None
    temp: Optional[float] = 25.0
    humidity: Optional[float] = 60.0
    windSpeed: Optional[float] = 5.0
    is_work: Optional[int] = None
    is_peak: Optional[int] = None
    code: Optional[int] = 999

class Fut(BaseModel):
    ts: datetime  # 改为 ts
    temp: Optional[float] = 25.0
    humidity: Optional[float] = 60.0
    windSpeed: Optional[float] = 5.0
    is_work: Optional[int] = None
    is_peak: Optional[int] = None
    code: Optional[int] = 999

class Req(BaseModel):
    station_id: str
    past_data: List[Past]  = Field(..., description=">=480 行历史")
    future_external: List[Fut] = Field(..., description="672 行未来")

class Item(BaseModel):
    ts: datetime  # 改为 ts
    forward_total_active_energy_pred: float  # 预测的用电量
    total_active_power_pred: float  # 预测的功率

class Resp(BaseModel):
    station_id: str
    model_used: str
    predictions: List[Item]

# ------------- Feature 工具 -------------
cn_holidays = holidays.country_holidays("CN")
def is_peak_vec(h, mi):
    return (((h > 8) | ((h == 8) & (mi >= 30))) &
            ((h < 17) | ((h == 17) & (mi <= 30)))).astype(int)

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    # 重命名以保持兼容性
    if 'ts' in df.columns:
        df = df.rename(columns={'ts': 'energy_date'})
    if 'forward_total_active_energy' in df.columns:
        df = df.rename(columns={'forward_total_active_energy': 'load_discharge_delta'})
    
    df["hour"]    = df["energy_date"].dt.hour
    df["minute"]  = df["energy_date"].dt.minute
    df["weekday"] = df["energy_date"].dt.weekday
    df["day"]     = df["energy_date"].dt.day
    df["month"]   = df["energy_date"].dt.month

    # 天气物理特征（如果有天气数据）
    if 'temp' in df.columns and 'humidity' in df.columns:
        df["dew_point"]  = df["temp"] - (100 - df["humidity"]) / 5
        df["feels_like"] = df["temp"] + 0.33 * df["humidity"] - 4
        df["temp_diff1"]  = df["temp"].diff(1)
        df["temp_diff24"] = df["temp"].diff(24)
    else:
        # 如果没有天气数据，创建虚拟特征
        df["dew_point"] = 20.0
        df["feels_like"] = 25.0
        df["temp_diff1"] = 0.0
        df["temp_diff24"] = 0.0

    df["sin_hour"] = np.sin(2*np.pi*(df["hour"]+df["minute"]/60)/24)
    df["cos_hour"] = np.cos(2*np.pi*(df["hour"]+df["minute"]/60)/24)
    df["sin_wday"] = np.sin(2*np.pi*df["weekday"]/7)
    df["cos_wday"] = np.cos(2*np.pi*df["weekday"]/7)

    df["is_holiday"] = df["energy_date"].isin(cn_holidays).astype(int)
    if "is_work" not in df.columns or df["is_work"].isna().any():
        df["is_work"] = ((df["weekday"]<5)&(~df["energy_date"].isin(cn_holidays))).astype(int)
    if "is_peak" not in df.columns or df["is_peak"].isna().any():
        df["is_peak"] = is_peak_vec(df["hour"], df["minute"])
    return df

def ensure_cols(df: pd.DataFrame, cols: list[str]):
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    return df

# 动态解析窗口
def parse_ws(prefix): 
    return sorted({int(c[len(prefix):]) for c in enc_cols if c.startswith(prefix)})
lag_ws  = parse_ws("load_lag")
ma_ws   = parse_ws("load_ma")
std_ws  = parse_ws("load_std")

# ---------------- Encoder ----------------
def build_encoder(past_df: pd.DataFrame) -> torch.Tensor:
    p = past_df.sort_values("ts").copy()
    p = enrich(p)

    # 滞后 / rolling
    for l in lag_ws:
        p[f"load_lag{l}"] = p["load_discharge_delta"].shift(l)
    for w in ma_ws:
        p[f"load_ma{w}"]  = p["load_discharge_delta"].rolling(w, min_periods=1).mean()
    for w in std_ws:
        p[f"load_std{w}"] = p["load_discharge_delta"].rolling(w, min_periods=1).std()

    p = p.ffill().bfill().fillna(0)

    if len(p) < PAST_STEPS:
        raise ValueError(f"past_data 需 ≥{PAST_STEPS} 行")

    p = ensure_cols(p, enc_cols)
    enc_np = sc_enc.transform(p.tail(PAST_STEPS)[enc_cols])
    return torch.from_numpy(enc_np.astype(np.float32)).unsqueeze(0)

# ---------------- Decoder ----------------
def build_decoder(fut_df: pd.DataFrame, last_load: float) -> torch.Tensor:
    f = fut_df.sort_values("ts").copy()
    if len(f) != FUTURE_STEPS:
        raise ValueError(f"future_external 需 {FUTURE_STEPS} 行")

    f = enrich(f)
    if "prev_load" in dec_cols:
        f["prev_load"] = last_load

    # code one-hot
    if any(c.startswith("code_") for c in dec_cols):
        f["code"] = f["code"].fillna(999).astype(int)
        for c in dec_cols:
            if c.startswith("code_"):
                val = int(c.split("_")[1])
                f[c] = (f["code"] == val).astype(int)

    f = f.ffill().bfill().fillna(0)
    f = ensure_cols(f, dec_cols)
    dec_np = sc_dec.transform(f[dec_cols])
    return torch.from_numpy(dec_np.astype(np.float32)).unsqueeze(0)

# ------------- 模型缓存 -------------
def load_model(station_id: str):
    if station_id in model_cache:
        return model_cache[station_id], True
    p = f"{ROOT}/model_{station_id}/model_optimized_{station_id}.pth"
    if os.path.exists(p):
        m = EncDec(len(enc_cols), len(dec_cols)).to(DEVICE)
        m.load_state_dict(torch.load(p, map_location=DEVICE)); m.eval()
        model_cache[station_id] = m
        return m, True
    return base_model, False

# ------------- FastAPI -------------
app = FastAPI(title="7-Day Load Forecast API")

@app.post("/predict", response_model=Resp)
def predict(req: Req):
    past_df = pd.DataFrame([x.dict() for x in req.past_data])
    fut_df  = pd.DataFrame([x.dict() for x in req.future_external])

    try:
        xe = build_encoder(past_df).to(DEVICE)
        # 使用新字段名获取最后的负荷值
        last_ld = past_df.sort_values("ts")["forward_total_active_energy"].iloc[-1]
        xd = build_decoder(fut_df, last_ld).to(DEVICE)
    except ValueError as e:
        raise HTTPException(400, str(e))

    model, flag = load_model(req.station_id)
    with torch.no_grad():
        y_energy_scaled, y_power_scaled = model(xe, xd)
        y_energy_scaled = y_energy_scaled.cpu().numpy().flatten()
        y_power_scaled = y_power_scaled.cpu().numpy().flatten()

    # 反标准化得到预测结果
    y_energy = sc_y_energy.inverse_transform(y_energy_scaled.reshape(-1,1)).flatten()
    y_power = sc_y_power.inverse_transform(y_power_scaled.reshape(-1,1)).flatten()
    
    # 清洗 nan / inf
    bad_energy = np.sum(~np.isfinite(y_energy))
    bad_power = np.sum(~np.isfinite(y_power))
    if bad_energy or bad_power:
        print(f"[WARN] 非有限预测值 - Energy: {bad_energy} 个, Power: {bad_power} 个，已替换为 0/±1e9")
    y_energy = np.nan_to_num(y_energy, nan=0.0, posinf=1e9, neginf=-1e9)
    y_power = np.nan_to_num(y_power, nan=0.0, posinf=1e9, neginf=-1e9)

    preds = [
        Item(
            ts=fut_df["ts"].iloc[i],
            forward_total_active_energy_pred=float(y_energy[i]),
            total_active_power_pred=float(y_power[i]),
        )
        for i in range(len(y_energy))
    ]
    
    return Resp(
        station_id=req.station_id,
        model_used="station" if flag else "base",
        predictions=preds,
    )

# ------------- main -------------
if __name__ == "__main__":
    uvicorn.run("api_encdec_7d:app", host="0.0.0.0", port=8000)