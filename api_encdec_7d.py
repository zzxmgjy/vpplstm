"""
api_encdec_7d.py  —  FastAPI 服务
上传字段只需: energy_date, load_discharge_delta, temp, humidity, windSpeed,
            is_work, is_peak, code
所有衍生列由服务端自动计算
"""
import os, joblib, holidays, uvicorn, torch, torch.nn as nn
import pandas as pd, numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
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
sc_y    = joblib.load(f"{ROOT}/scaler_y.pkl")

# ------------ LSTM EncDec ------------
class EncDec(nn.Module):
    def __init__(self, d_enc, d_dec, hid=128, drop=.24):
        super().__init__()
        self.enc = nn.LSTM(d_enc, hid, batch_first=True)
        self.dec = nn.LSTM(d_dec, hid, batch_first=True)
        self.dp  = nn.Dropout(drop)
        self.fc  = nn.Linear(hid, 1)
    def forward(self, xe, xd):
        _, (h, c) = self.enc(xe)
        out, _    = self.dec(xd, (h, c))
        return self.fc(self.dp(out)).squeeze(-1)

base_model = EncDec(len(enc_cols), len(dec_cols)).to(DEVICE)
base_model.load_state_dict(torch.load(f"{ROOT}/model_weighted.pth", map_location=DEVICE))
base_model.eval()
model_cache = {}

# ------------- Pydantic schema -------------
class Past(BaseModel):
    energy_date: datetime
    load_discharge_delta: float
    temp: float; humidity: float; windSpeed: float
    is_work: int | None = None
    is_peak: int | None = None
    code: int | None = 999
class Fut(BaseModel):
    energy_date: datetime
    temp: float; humidity: float; windSpeed: float
    is_work: int | None = None
    is_peak: int | None = None
    code: int | None = 999
class Req(BaseModel):
    station_id: str
    past_data: List[Past]  = Field(..., description=">=480 行历史")
    future_external: List[Fut] = Field(..., description="672 行未来")
class Item(BaseModel):
    energy_date: datetime; load_discharge_delta_pred: float
class Resp(BaseModel):
    station_id: str; model_used: str; predictions: List[Item]

# ------------- Feature 工具 -------------
cn_holidays = holidays.country_holidays("CN")
def is_peak_vec(h, mi):
    return (((h > 8) | ((h == 8) & (mi >= 30))) &
            ((h < 17) | ((h == 17) & (mi <= 30)))).astype(int)

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"]    = df["energy_date"].dt.hour
    df["minute"]  = df["energy_date"].dt.minute
    df["weekday"] = df["energy_date"].dt.weekday
    df["day"]     = df["energy_date"].dt.day
    df["month"]   = df["energy_date"].dt.month

    df["temp_squared"] = df["temp"] ** 2
    df["dew_point"]  = df["temp"] - (100 - df["humidity"]) / 5
    df["feels_like"] = df["temp"] + 0.33 * df["humidity"] - 4
    df["temp_diff1"]  = df["temp"].diff(1)
    df["temp_diff24"] = df["temp"].diff(24)

    df["sin_hour"] = np.sin(2*np.pi*(df["hour"]+df["minute"]/60)/24)
    df["cos_hour"] = np.cos(2*np.pi*(df["hour"]+df["minute"]/60)/24)
    df["sin_wday"] = np.sin(2*np.pi*df["weekday"]/7)
    df["cos_wday"] = np.cos(2*np.pi*df["weekday"]/7)

    df["is_holiday"] = df["energy_date"].isin(cn_holidays).astype(int)
    if "is_work" not in df.columns:
        df["is_work"] = ((df["weekday"]<5)&(~df["energy_date"].isin(cn_holidays))).astype(int)
    if "is_peak" not in df.columns:
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
    p = past_df.sort_values("energy_date").copy()
    p = enrich(p)

    # 滞后 / rolling
    for l in lag_ws:
        p[f"load_lag{l}"] = p["load_discharge_delta"].shift(l)
    for w in ma_ws:
        p[f"load_ma{w}"]  = p["load_discharge_delta"].rolling(w, min_periods=1).mean()
    for w in std_ws:
        p[f"load_std{w}"] = p["load_discharge_delta"].rolling(w, min_periods=1).std()

    # ----------- 关键改动 ▲-----------
    p = p.ffill().bfill().fillna(0)
    # ----------------------------------

    if len(p) < PAST_STEPS:
        raise ValueError(f"past_data 需 ≥{PAST_STEPS} 行")

    p = ensure_cols(p, enc_cols)
    enc_np = sc_enc.transform(p.tail(PAST_STEPS)[enc_cols])
    return torch.from_numpy(enc_np.astype(np.float32)).unsqueeze(0)

# ---------------- Decoder ----------------
def build_decoder(fut_df: pd.DataFrame, last_load: float) -> torch.Tensor:
    f = fut_df.sort_values("energy_date").copy()
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

    # ----------- 关键改动 ▲-----------
    f = f.ffill().bfill().fillna(0)
    # ----------------------------------

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
        last_ld = past_df.sort_values("energy_date")["load_discharge_delta"].iloc[-1]
        xd = build_decoder(fut_df, last_ld).to(DEVICE)
    except ValueError as e:
        raise HTTPException(400, str(e))

    model, flag = load_model(req.station_id)
    with torch.no_grad():
        y_scaled = model(xe, xd).cpu().numpy().flatten()

    # ---------- 新增：清洗 nan / inf ----------
    y = sc_y.inverse_transform(y_scaled.reshape(-1,1)).flatten()
    bad = np.sum(~np.isfinite(y))
    if bad:
        print(f"[WARN] 非有限预测值 {bad} 个，已替换为 0/±1e9")
    y = np.nan_to_num(y, nan=0.0, posinf=1e9, neginf=-1e9)
    # ------------------------------------------

    preds = [
        Item(
            energy_date=fut_df["energy_date"].iloc[i],
            load_discharge_delta_pred=float(v),
        )
        for i, v in enumerate(y)
    ]
    return Resp(
        station_id=req.station_id,
        model_used="station" if flag else "base",
        predictions=preds,
    )

# ------------- main -------------
if __name__ == "__main__":
    uvicorn.run("api_encdec_7d:app", host="0.0.0.0", port=8000)