# fine_tune_station.py
# ==========================================================
# 单站微调脚本（无需命令行参数）
# ----------------------------------------------------------
#  使用前修改： STATION_ID / CSV_FILE / EPOCHS / BATCH_SIZE
# ==========================================================

import os,argparse, warnings, holidays, joblib
import pandas as pd, numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
warnings.filterwarnings("ignore")

# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument('--station_id', required=True)
parser.add_argument('--file',       required=True)
args = parser.parse_args()
# -------- 用户配置 --------
STATION_ID  = args.station_id           # ← 场站 ID
CSV_FILE    = args.file # ← 该场站 CSV
EPOCHS      = 40
BATCH_SIZE  = 128
# --------------------------

ROOT       = 'output_pytorch'
PAST_STEPS = 96*5
FUT_STEPS  = 96*7

# ---------- is_peak 08:30~17:30 ----------
def make_is_peak(ts):
    h  = ts.dt.hour
    mi = ts.dt.minute
    s  = (h > 8) | ((h == 8) & (mi >= 30))
    e  = (h < 17) | ((h == 17) & (mi <= 30))
    return (s & e).astype(int)

# ---------- 读取数据 ----------
df = pd.read_csv(CSV_FILE, parse_dates=['energy_date'])
if 'station_ref_id' in df.columns:
    df = df[df['station_ref_id'] == STATION_ID]
df = df.sort_values('energy_date')
if df.empty:
    raise ValueError('CSV 中没有指定场站的数据')

# ---------- enrich 基础特征 ----------
cn_holidays = holidays.country_holidays('CN')
def enrich(d):
    d['hour']     = d['energy_date'].dt.hour
    d['minute']   = d['energy_date'].dt.minute
    d['weekday']  = d['energy_date'].dt.weekday
    d['temp_squared'] = d['temp']**2
    d['sin_hour'] = np.sin(2*np.pi*(d['hour']+d['minute']/60)/24)
    d['cos_hour'] = np.cos(2*np.pi*(d['hour']+d['minute']/60)/24)
    d['sin_wday'] = np.sin(2*np.pi*d['weekday']/7)
    d['cos_wday'] = np.cos(2*np.pi*d['weekday']/7)
    d['is_holiday'] = d['energy_date'].isin(cn_holidays).astype(int)
    if 'is_work' not in d.columns:
        d['is_work'] = ((d['weekday'] < 5) & (d['is_holiday'] == 0)).astype(int)
    if 'is_peak' not in d.columns:
        d['is_peak'] = make_is_peak(d['energy_date'])
    return d
df = enrich(df)

# ---------- 负荷衍生 ----------
for lag in [1,4,24,96]:
    df[f'load_lag{lag}'] = df['load_discharge_delta'].shift(lag)
for w in [4,24,96]:
    df[f'load_ma{w}']  = df['load_discharge_delta'].rolling(w,1).mean()
    df[f'load_std{w}'] = df['load_discharge_delta'].rolling(w,1).std()
df = df.fillna(method='ffill').dropna()
if len(df) < PAST_STEPS + FUT_STEPS:
    raise ValueError('数据量不足以微调')

# ---------- 特征列 ----------
ENC = [
    'temp','temp_squared','humidity','windSpeed',
    'load_lag1','load_lag4','load_lag24','load_lag96',
    'load_ma4','load_ma24','load_ma96',
    'load_std4','load_std24','load_std96',
    'is_holiday','is_work','is_peak',
    'sin_hour','cos_hour','sin_wday','cos_wday'
]
DEC = [
    'temp','humidity','windSpeed',
    'is_holiday','is_work','is_peak',
    'sin_hour','cos_hour','sin_wday','cos_wday'
]

# ---------- 载入 scaler ----------
sc_enc = joblib.load(f'{ROOT}/scaler_enc.pkl')
sc_dec = joblib.load(f'{ROOT}/scaler_dec.pkl')
sc_y   = joblib.load(f'{ROOT}/scaler_y.pkl')

# ---------- 制作训练样本 ----------
def make_data(ds):
    Xp, Xf, Ys = [], [], []
    enc = sc_enc.transform(ds[ENC])
    dec = sc_dec.transform(ds[DEC])
    y   = sc_y.transform(ds[['load_discharge_delta']])
    for i in range(len(ds) - PAST_STEPS - FUT_STEPS + 1):
        Xp.append(enc[i:i+PAST_STEPS])
        Xf.append(dec[i+PAST_STEPS:i+PAST_STEPS+FUT_STEPS])
        Ys.append(y[i+PAST_STEPS:i+PAST_STEPS+FUT_STEPS, 0])
    return (np.array(Xp, np.float32),
            np.array(Xf, np.float32),
            np.array(Ys, np.float32))

Xp, Xf, Y = make_data(df)
split = int(.8 * len(Xp))
tr_ds = TensorDataset(torch.from_numpy(Xp[:split]),
                      torch.from_numpy(Xf[:split]),
                      torch.from_numpy(Y[:split]))
va_ds = TensorDataset(torch.from_numpy(Xp[split:]),
                      torch.from_numpy(Xf[split:]),
                      torch.from_numpy(Y[split:]))
tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)

# ---------- 构建模型 ----------
class EncDec(nn.Module):
    def __init__(self, d_enc, d_dec, hid, fut, drop):
        super().__init__()
        self.enc = nn.LSTM(d_enc, hid, batch_first=True)
        self.dec = nn.LSTM(d_dec, hid, batch_first=True)
        self.dp  = nn.Dropout(drop)
        self.fc  = nn.Linear(hid, 1)
    def forward(self, xe, xd):
        _, (h, c) = self.enc(xe)
        out, _ = self.dec(xd, (h, c))
        return self.fc(self.dp(out)).squeeze(-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = EncDec(len(ENC), len(DEC), 128, FUT_STEPS, .24).to(device)
model.load_state_dict(torch.load(f'{ROOT}/model_base_encdec7d.pth', map_location=device))
for p in model.enc.parameters(): p.requires_grad_(False)   # 冻结 Encoder

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=.5)

best, patience = 1e9, 0
for ep in range(1, EPOCHS + 1):
    # ---- train ----
    model.train(); tr = 0
    for xe, xd, yy in tr_loader:
        xe, xd, yy = xe.to(device), xd.to(device), yy.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xe, xd), yy)
        loss.backward(); optimizer.step()
        tr += loss.item()
    tr /= len(tr_loader)

    # ---- valid ----
    model.eval(); va = 0
    with torch.no_grad():
        for xe, xd, yy in va_loader:
            xe, xd, yy = xe.to(device), xd.to(device), yy.to(device)
            va += criterion(model(xe, xd), yy).item()
    va /= len(va_loader)
    scheduler.step(va)

    log = f'E{ep:03d}  tr {tr:.4f}  va {va:.4f}'
    if va < best:
        best = va; patience = 0
        folder = os.path.join(ROOT, f'mode_{STATION_ID}'); os.makedirs(folder, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(folder, f'model_optimized_{STATION_ID}.pth'))
        log += ' ✔ save'
    else:
        patience += 1
        if patience >= 10:
            log += ' (early stop)'
            print(log); break
    print(log)

# ---------- Quick MAPE / RMSE ----------
model.eval()
with torch.no_grad():
    xe = torch.from_numpy(Xp[-1:].astype(np.float32)).to(device)
    xd = torch.from_numpy(Xf[-1:].astype(np.float32)).to(device)
    pred_scaled = model(xe, xd).cpu().numpy().flatten()
    pred = sc_y.inverse_transform(pred_scaled.reshape(-1,1)).flatten()
    true = sc_y.inverse_transform(Y[-1:].reshape(-1,1)).flatten()
    true_safe = np.where(true == 0, 1e-6, true)
    mape = mean_absolute_percentage_error(true_safe, pred) * 100
    rmse = np.sqrt(mean_squared_error(true, pred))
    print(f'\n【{STATION_ID}】 quick-MAPE = {mape:.2f}%   RMSE = {rmse:.2f}')

print('\n微调完成 ✅  模型保存至 output_pytorch/mode_<ID>/model_optimized_<ID>.pth')