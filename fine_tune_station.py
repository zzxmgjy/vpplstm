# fine_tune_station.py
# ==========================================================
# 单站微调（使用 WeightedL1 + prev_load Teacher-Forcing）
# ==========================================================
import os, warnings, holidays, joblib
import pandas as pd, numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
warnings.filterwarnings("ignore")

# ---------- 用户配置 ----------
STATION_ID  = 3205103743359       # ← 修改为目标场站
CSV_FILE    = 'loaddata.csv'      # ← 该场站 CSV
EPOCHS      = 50
BATCH_SIZE  = 128
# --------------------------------

ROOT        = 'output_pytorch'
PAST_STEPS  = 96*5
FUT_STEPS   = 96*7                # 672

# ---------- is_peak ----------
def make_is_peak(ts):
    h, mi = ts.dt.hour, ts.dt.minute
    s  = (h > 8) | ((h == 8) & (mi >= 30))
    e  = (h < 17) | ((h == 17) & (mi <= 30))
    return (s & e).astype(int)

# ---------- 读数据 ----------
df = pd.read_csv(CSV_FILE, parse_dates=['energy_date'])
if 'station_ref_id' in df.columns:
    df = df[df['station_ref_id'] == STATION_ID]
df = df.sort_values('energy_date')
if df.empty:
    raise ValueError('CSV 中没有指定场站的数据')

# ---------- enrich（保持与总模型一致） ----------
cn_holidays = holidays.country_holidays('CN')
def enrich(d):
    d['hour']    = d['energy_date'].dt.hour
    d['minute']  = d['energy_date'].dt.minute
    d['weekday'] = d['energy_date'].dt.weekday
    d['month']   = d['energy_date'].dt.month
    d['day']     = d['energy_date'].dt.day

    d['dew_point']  = d['temp'] - (100-d['humidity'])/5
    d['feels_like'] = d['temp'] + 0.33*d['humidity'] - 4
    for k in [1,24]:
        d[f'temp_diff{k}'] = d['temp'].diff(k)

    d['sin_hour'] = np.sin(2*np.pi*(d['hour']+d['minute']/60)/24)
    d['cos_hour'] = np.cos(2*np.pi*(d['hour']+d['minute']/60)/24)
    d['sin_wday'] = np.sin(2*np.pi*d['weekday']/7)
    d['cos_wday'] = np.cos(2*np.pi*d['weekday']/7)

    d['is_holiday'] = d['energy_date'].isin(cn_holidays).astype(int)
    d['is_work']    = ((d['weekday']<5)&(~d['energy_date'].isin(cn_holidays))).astype(int)
    d['is_peak']    = make_is_peak(d['energy_date']).astype(int)
    return d
df = enrich(df)

# ---------- 滞后 / rolling（与总模型保持一致即可） ----------
for lag in [1,2,4,8,12,24,48,96]:
    df[f'load_lag{lag}'] = df['load_discharge_delta'].shift(lag)
for w in [4,8,12,24,48,96]:
    df[f'load_ma{w}']  = df['load_discharge_delta'].rolling(w,1).mean()
    df[f'load_std{w}'] = df['load_discharge_delta'].rolling(w,1).std()

# prev_load (teacher forcing)
df['prev_load'] = df['load_discharge_delta'].shift(1)

df = df.fillna(method='ffill').fillna(method='bfill').dropna()
if len(df) < PAST_STEPS + FUT_STEPS:
    raise ValueError('数据量不足以微调')

# ---------- 载入  enc_cols / dec_cols & scaler ----------
enc_cols = joblib.load(f'{ROOT}/enc_cols.pkl')
dec_cols = joblib.load(f'{ROOT}/dec_cols.pkl')
sc_enc   = joblib.load(f'{ROOT}/scaler_enc.pkl')
sc_dec   = joblib.load(f'{ROOT}/scaler_dec.pkl')
sc_y     = joblib.load(f'{ROOT}/scaler_y.pkl')

# 若缺少列则补 0，确保维度一致
for col in set(enc_cols + dec_cols):
    if col not in df.columns:
        df[col] = 0

# ---------- 制作滑窗 ----------
def make_ds(data):
    Xp, Xf, Ys = [], [], []
    enc = sc_enc.transform(data[enc_cols])
    dec = sc_dec.transform(data[dec_cols])
    y   = sc_y.transform(data[['load_discharge_delta']])
    for i in range(len(data) - PAST_STEPS - FUT_STEPS + 1):
        Xp.append(enc[i:i+PAST_STEPS])
        Xf.append(dec[i+PAST_STEPS:i+PAST_STEPS+FUT_STEPS])
        Ys.append(y[i+PAST_STEPS:i+PAST_STEPS+FUT_STEPS, 0])
    return (np.array(Xp, np.float32),
            np.array(Xf, np.float32),
            np.array(Ys, np.float32))

Xp, Xf, Y = make_ds(df)
split = int(.8*len(Xp))
tr_ds = TensorDataset(torch.from_numpy(Xp[:split]),
                      torch.from_numpy(Xf[:split]),
                      torch.from_numpy(Y[:split]))
va_ds = TensorDataset(torch.from_numpy(Xp[split:]),
                      torch.from_numpy(Xf[split:]),
                      torch.from_numpy(Y[split:]))
tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)

# ---------- 模型结构（与总模型一致） ----------
class EncDec(nn.Module):
    def __init__(self, d_enc, d_dec, hid, drop):
        super().__init__()
        self.enc = nn.LSTM(d_enc, hid, batch_first=True)
        self.dec = nn.LSTM(d_dec, hid, batch_first=True)
        self.dp  = nn.Dropout(drop)
        self.fc  = nn.Linear(hid, 1)
    def forward(self, xe, xd):
        _, (h,c) = self.enc(xe)
        out,_ = self.dec(xd,(h,c))
        return self.fc(self.dp(out)).squeeze(-1)

# ---------- 加权 L1  ----------
class WeightedL1(nn.Module):
    def __init__(self, fut, device):
        super().__init__()
        w = np.concatenate([
            np.ones(96*2),        # Day1-2
            np.ones(96)*1.3,      # Day3
            np.ones(96)*1.5,      # Day4
            np.ones(96*3)*1.2     # Day5-7
        ])
        self.register_buffer("w", torch.tensor(w, dtype=torch.float32, device=device))
    def forward(self, pred, target):
        return torch.mean(self.w * torch.abs(pred - target))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = EncDec(len(enc_cols), len(dec_cols), 128, .24).to(device)
model.load_state_dict(torch.load(f'{ROOT}/model_weighted.pth', map_location=device))

# ------ 只微调 Decoder + FC ------
for p in model.enc.parameters():
    p.requires_grad_(False)

criterion = WeightedL1(FUT_STEPS, device)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=.5)

best, wait = 1e9, 0
print(f"\n🚀 开始微调  station={STATION_ID}  samples={len(tr_ds)} / {len(va_ds)}")
for ep in range(1, EPOCHS+1):
    model.train(); tr=0
    for xe,xd,yy in tr_loader:
        xe,xd,yy = xe.to(device), xd.to(device), yy.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xe,xd), yy)
        loss.backward(); optimizer.step()
        tr += loss.item()
    tr /= len(tr_loader)

    model.eval(); va=0
    with torch.no_grad():
        for xe,xd,yy in va_loader:
            xe,xd,yy = xe.to(device), xd.to(device), yy.to(device)
            va += criterion(model(xe,xd), yy).item()
    va /= len(va_loader); scheduler.step(va)

    log = f'E{ep:03d}  tr {tr:.4f}  va {va:.4f}'
    if va < best:
        best = va; wait = 0
        folder = f'{ROOT}/model_{STATION_ID}'; os.makedirs(folder, exist_ok=True)
        torch.save(model.state_dict(), f'{folder}/model_optimized_{STATION_ID}.pth')
        log += '  ✔ save'
    else:
        wait += 1
        if wait >= 10:
            log += '  (early stop)'; print(log); break
    print(log)

# ---------- 评估 ----------
def day_mape(t,p):
    res=[]
    for d in range(7):
        s,e=d*96,(d+1)*96
        t0,t1=t[s:e],p[s:e]
        t0=np.where(t0==0,1e-6,t0)
        res.append(mean_absolute_percentage_error(t0,t1)*100)
    return res

model.eval()
with torch.no_grad():
    xe = torch.from_numpy(Xp[-1:].astype(np.float32)).to(device)
    xd = torch.from_numpy(Xf[-1:].astype(np.float32)).to(device)
    pred_s = model(xe,xd).cpu().numpy().flatten()
    pred   = sc_y.inverse_transform(pred_s.reshape(-1,1)).flatten()
    true   = sc_y.inverse_transform(Y[-1:].reshape(-1,1)).flatten()

    true_safe = np.where(true==0,1e-6,true)
    mape_all  = mean_absolute_percentage_error(true_safe,pred)*100
    rmse_all  = np.sqrt(mean_squared_error(true,pred))

    dm = day_mape(true,pred)
    dm_str = ' | '.join([f'D{i+1}:{m:.2f}%' for i,m in enumerate(dm)])

print(f'\n📊 【{STATION_ID}】7-day MAPE={mape_all:.2f}%  RMSE={rmse_all:.2f}')
print(f'📊 【{STATION_ID}】Day-wise MAPE  {dm_str}')
print(f'\n✅ 微调完成  模型已保存至 output_pytorch/model_{STATION_ID}/model_optimized_{STATION_ID}.pth')