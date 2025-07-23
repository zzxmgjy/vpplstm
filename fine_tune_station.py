# fine_tune_station.py
# ==========================================================
# 单站微调（使用 WeightedL1 + prev_load Teacher-Forcing）
# Updated for vpp_meter.csv with dual outputs (energy + power)
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
CSV_FILE    = 'vpp_meter.csv'    # ← 使用 vpp_meter.csv
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
df = pd.read_csv(CSV_FILE, parse_dates=['ts'])
if 'station_ref_id' in df.columns:
    df = df[df['station_ref_id'] == STATION_ID]
df = df.sort_values('ts')
if df.empty:
    raise ValueError('CSV 中没有指定场站的数据')

# 重命名字段以保持代码兼容性
df = df.rename(columns={
    'ts': 'energy_date',
    'forward_total_active_energy': 'load_discharge_delta'
})

# 处理可能缺失的字段，设置默认值
optional_fields = {
    'temp': 25.0,           # 默认温度25度
    'humidity': 60.0,       # 默认湿度60%
    'windSpeed': 5.0,       # 默认风速5m/s
    'is_work': None,        # 将根据日期计算
    'is_peak': None,        # 将根据时间计算
    'code': 999             # 默认代码
}

for field, default_value in optional_fields.items():
    if field not in df.columns:
        if field in ['is_work', 'is_peak']:
            df[field] = 0  # 临时设置，后面会重新计算
        else:
            df[field] = default_value
        print(f"⚠️  字段 '{field}' 缺失，已设置默认值: {default_value}")

# ---------- enrich（保持与总模型一致） ----------
cn_holidays = holidays.country_holidays('CN')
def enrich(d):
    d['hour']    = d['energy_date'].dt.hour
    d['minute']  = d['energy_date'].dt.minute
    d['weekday'] = d['energy_date'].dt.weekday
    d['month']   = d['energy_date'].dt.month
    d['day']     = d['energy_date'].dt.day

    # 天气物理特征（如果有天气数据）
    if 'temp' in d.columns and 'humidity' in d.columns:
        d['dew_point']  = d['temp'] - (100-d['humidity'])/5
        d['feels_like'] = d['temp'] + 0.33*d['humidity'] - 4
        for k in [1,24]:
            d[f'temp_diff{k}'] = d['temp'].diff(k)
    else:
        # 如果没有天气数据，创建虚拟特征
        d['dew_point'] = 20.0
        d['feels_like'] = 25.0
        d['temp_diff1'] = 0.0
        d['temp_diff24'] = 0.0

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
sc_y_energy = joblib.load(f'{ROOT}/scaler_y_energy.pkl')
sc_y_power  = joblib.load(f'{ROOT}/scaler_y_power.pkl')

# 若缺少列则补 0，确保维度一致
for col in set(enc_cols + dec_cols):
    if col not in df.columns:
        df[col] = 0

# ---------- 制作滑窗 - 双输出 ----------
def make_ds(data):
    Xp, Xf, Y_energy, Y_power = [], [], [], []
    enc = sc_enc.transform(data[enc_cols])
    dec = sc_dec.transform(data[dec_cols])
    y_energy = sc_y_energy.transform(data[['load_discharge_delta']])
    y_power  = sc_y_power.transform(data[['total_active_power']])
    for i in range(len(data) - PAST_STEPS - FUT_STEPS + 1):
        Xp.append(enc[i:i+PAST_STEPS])
        Xf.append(dec[i+PAST_STEPS:i+PAST_STEPS+FUT_STEPS])
        Y_energy.append(y_energy[i+PAST_STEPS:i+PAST_STEPS+FUT_STEPS, 0])
        Y_power.append(y_power[i+PAST_STEPS:i+PAST_STEPS+FUT_STEPS, 0])
    return (np.array(Xp, np.float32),
            np.array(Xf, np.float32),
            np.array(Y_energy, np.float32),
            np.array(Y_power, np.float32))

Xp, Xf, Y_energy, Y_power = make_ds(df)
split = int(.8*len(Xp))
tr_ds = TensorDataset(torch.from_numpy(Xp[:split]),
                      torch.from_numpy(Xf[:split]),
                      torch.from_numpy(Y_energy[:split]),
                      torch.from_numpy(Y_power[:split]))
va_ds = TensorDataset(torch.from_numpy(Xp[split:]),
                      torch.from_numpy(Xf[split:]),
                      torch.from_numpy(Y_energy[split:]),
                      torch.from_numpy(Y_power[split:]))
tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)

# ---------- 模型结构（与总模型一致） - 双输出 ----------
class EncDec(nn.Module):
    def __init__(self, d_enc, d_dec, hid, drop):
        super().__init__()
        self.enc = nn.LSTM(d_enc, hid, batch_first=True)
        self.dec = nn.LSTM(d_dec, hid, batch_first=True)
        self.dp  = nn.Dropout(drop)
        self.fc_energy = nn.Linear(hid, 1)  # 电量预测
        self.fc_power  = nn.Linear(hid, 1)  # 功率预测
    def forward(self, xe, xd):
        _, (h,c) = self.enc(xe)
        out,_ = self.dec(xd,(h,c))
        out_dp = self.dp(out)
        energy_pred = self.fc_energy(out_dp).squeeze(-1)
        power_pred = self.fc_power(out_dp).squeeze(-1)
        return energy_pred, power_pred

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
    for xe,xd,yy_energy,yy_power in tr_loader:
        xe,xd,yy_energy,yy_power = xe.to(device), xd.to(device), yy_energy.to(device), yy_power.to(device)
        optimizer.zero_grad()
        pred_energy, pred_power = model(xe,xd)
        loss_energy = criterion(pred_energy, yy_energy)
        loss_power = criterion(pred_power, yy_power)
        loss = loss_energy + loss_power
        loss.backward(); optimizer.step()
        tr += loss.item()
    tr /= len(tr_loader)

    model.eval(); va=0
    with torch.no_grad():
        for xe,xd,yy_energy,yy_power in va_loader:
            xe,xd,yy_energy,yy_power = xe.to(device), xd.to(device), yy_energy.to(device), yy_power.to(device)
            pred_energy, pred_power = model(xe,xd)
            loss_energy = criterion(pred_energy, yy_energy)
            loss_power = criterion(pred_power, yy_power)
            va += (loss_energy + loss_power).item()
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

# ---------- 评估 - 双输出 ----------
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
    pred_energy_s, pred_power_s = model(xe,xd)
    pred_energy_s = pred_energy_s.cpu().numpy().flatten()
    pred_power_s = pred_power_s.cpu().numpy().flatten()
    
    pred_energy = sc_y_energy.inverse_transform(pred_energy_s.reshape(-1,1)).flatten()
    pred_power = sc_y_power.inverse_transform(pred_power_s.reshape(-1,1)).flatten()
    
    true_energy = sc_y_energy.inverse_transform(Y_energy[-1:].reshape(-1,1)).flatten()
    true_power = sc_y_power.inverse_transform(Y_power[-1:].reshape(-1,1)).flatten()

    true_energy_safe = np.where(true_energy==0,1e-6,true_energy)
    true_power_safe = np.where(true_power==0,1e-6,true_power)
    
    mape_energy = mean_absolute_percentage_error(true_energy_safe,pred_energy)*100
    mape_power = mean_absolute_percentage_error(true_power_safe,pred_power)*100
    
    rmse_energy = np.sqrt(mean_squared_error(true_energy,pred_energy))
    rmse_power = np.sqrt(mean_squared_error(true_power,pred_power))

    dm_energy = day_mape(true_energy,pred_energy)
    dm_power = day_mape(true_power,pred_power)
    
    dm_energy_str = ' | '.join([f'D{i+1}:{m:.2f}%' for i,m in enumerate(dm_energy)])
    dm_power_str = ' | '.join([f'D{i+1}:{m:.2f}%' for i,m in enumerate(dm_power)])

print(f'\n📊 【{STATION_ID}】Energy 7-day MAPE={mape_energy:.2f}%  RMSE={rmse_energy:.2f}')
print(f'📊 【{STATION_ID}】Power  7-day MAPE={mape_power:.2f}%   RMSE={rmse_power:.2f}')
print(f'📊 【{STATION_ID}】Energy Day-wise MAPE  {dm_energy_str}')
print(f'📊 【{STATION_ID}】Power  Day-wise MAPE  {dm_power_str}')
print(f'\n✅ 微调完成  模型已保存至 output_pytorch/model_{STATION_ID}/model_optimized_{STATION_ID}.pth')