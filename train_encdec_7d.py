# =========================================================
# 基础 Encoder–Decoder (7 天)  —  is_peak: 08:30~17:30
# =========================================================
import os, warnings, holidays, joblib
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

warnings.filterwarnings("ignore")
os.makedirs("output_pytorch", exist_ok=True)

CFG = dict(
    past_steps   = 96*5,
    future_steps = 96*7,
    hidden_dim   = 128,
    drop_rate    = .24,
    batch_size   = 128,
    epochs       = 200,
    patience     = 20,
    lr           = 2e-4
)

# ---------- is_peak 08:30 ~ 17:30 ----------
def make_is_peak(ts):
    h  = ts.dt.hour
    mi = ts.dt.minute
    cond_start = (h > 8) | ((h == 8)  & (mi >= 30))
    cond_end   = (h < 17) | ((h == 17) & (mi <= 30))
    return (cond_start & cond_end).astype(int)

# ---------- 1. 数据 ----------
df = pd.read_csv('loaddata.csv', parse_dates=['energy_date'])
df = df.sort_values(['station_ref_id', 'energy_date'])
cn_holidays = holidays.country_holidays('CN')

# ---------- 2. 基础特征 ----------
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

# ---------- 3. 负荷衍生（Encoder 专用） ----------
for lag in [1,4,24,96]:
    df[f'load_lag{lag}'] = df.groupby('station_ref_id')['load_discharge_delta'].shift(lag)
for w in [4,24,96]:
    g = df.groupby('station_ref_id')['load_discharge_delta']
    df[f'load_ma{w}']  = g.rolling(w,1).mean().reset_index(level=0, drop=True)
    df[f'load_std{w}'] = g.rolling(w,1).std().reset_index(level=0, drop=True)

df = df.fillna(method='ffill').dropna()

# ---------- 4. 特征列 ----------
ENC_COLS = [
    'temp','temp_squared','humidity','windSpeed',
    'load_lag1','load_lag4','load_lag24','load_lag96',
    'load_ma4','load_ma24','load_ma96',
    'load_std4','load_std24','load_std96',
    'is_holiday','is_work','is_peak',
    'sin_hour','cos_hour','sin_wday','cos_wday'
]
DEC_COLS = [
    'temp','humidity','windSpeed',
    'is_holiday','is_work','is_peak',
    'sin_hour','cos_hour','sin_wday','cos_wday'
]

# ---------- 5. 滑窗 ----------
def make_dataset(data, past, fut):
    Xp, Xf, Ys = [], [], []
    sc_enc, sc_dec, sc_y = StandardScaler(), StandardScaler(), StandardScaler()
    enc_all = sc_enc.fit_transform(data[ENC_COLS])
    dec_all = sc_dec.fit_transform(data[DEC_COLS])
    y_all   = sc_y.fit_transform(data[['load_discharge_delta']])

    enc_df = pd.DataFrame(enc_all, columns=ENC_COLS, index=data.index)
    enc_df['y'] = y_all
    dec_df = pd.DataFrame(dec_all, columns=DEC_COLS, index=data.index)

    for sid in data['station_ref_id'].unique():
        msk = data['station_ref_id'] == sid
        if msk.sum() < past + fut: continue
        e = enc_df[msk][ENC_COLS].values
        d = dec_df[msk][DEC_COLS].values
        y = enc_df[msk]['y'].values
        for i in range(len(e) - past - fut + 1):
            Xp.append(e[i:i+past])
            Xf.append(d[i+past:i+past+fut])
            Ys.append(y[i+past:i+past+fut])
    return np.array(Xp, np.float32), np.array(Xf, np.float32), np.array(Ys, np.float32), sc_enc, sc_dec, sc_y

Xp, Xf, Y, sc_enc, sc_dec, sc_y = make_dataset(df, CFG['past_steps'], CFG['future_steps'])
split = int(.8 * len(Xp))
train_ds = TensorDataset(torch.from_numpy(Xp[:split]), torch.from_numpy(Xf[:split]), torch.from_numpy(Y[:split]))
val_ds   = TensorDataset(torch.from_numpy(Xp[split:]), torch.from_numpy(Xf[split:]), torch.from_numpy(Y[split:]))
train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=CFG['batch_size'], shuffle=False)

# ---------- 6. 网络 ----------
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
        return self.fc(self.dp(out)).squeeze(-1)   # (B, fut)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = EncDec(len(ENC_COLS), len(DEC_COLS), CFG['hidden_dim'],
                CFG['future_steps'], CFG['drop_rate']).to(device)
crit   = nn.L1Loss()
opt    = torch.optim.Adam(model.parameters(), lr=CFG['lr'])
sch    = ReduceLROnPlateau(opt, 'min', patience=5, factor=.5)

best = 1e9; patience = 0
for epoch in range(1, CFG['epochs'] + 1):
    model.train(); tr_loss = 0
    for xe, xd, yy in train_loader:
        xe, xd, yy = xe.to(device), xd.to(device), yy.to(device)
        opt.zero_grad(); loss = crit(model(xe, xd), yy)
        loss.backward(); opt.step()
        tr_loss += loss.item()
    tr_loss /= len(train_loader)

    model.eval(); val_loss = 0
    with torch.no_grad():
        for xe, xd, yy in val_loader:
            xe, xd, yy = xe.to(device), xd.to(device), yy.to(device)
            val_loss += crit(model(xe, xd), yy).item()
    val_loss /= len(val_loader)
    sch.step(val_loss)

    print(f'E{epoch:03d}  tr {tr_loss:.4f}  va {val_loss:.4f}')
    if val_loss < best:
        best = val_loss; patience = 0
        torch.save(model.state_dict(), 'output_pytorch/model_base_encdec7d.pth')
        print('   ↳ save best')
    else:
        patience += 1
        if patience >= CFG['patience']:
            print('EarlyStop'); break

# ---------- 7. 保存 Scaler ----------
joblib.dump(sc_enc, 'output_pytorch/scaler_enc.pkl')
joblib.dump(sc_dec, 'output_pytorch/scaler_dec.pkl')
joblib.dump(sc_y  , 'output_pytorch/scaler_y.pkl')

# ---------- 8. 快速评估（新增每日 MAPE） ----------
def calc_day_mape(true_arr, pred_arr):
    """把 672 个点按 96 切 7 段，返回长度 7 的 MAPE 列表"""
    mape_list = []
    for d in range(7):
        s, e = d*96, (d+1)*96
        t_slice = true_arr[s:e]
        p_slice = pred_arr[s:e]
        t_safe  = np.where(t_slice==0, 1e-6, t_slice)
        mape_list.append(mean_absolute_percentage_error(t_safe, p_slice)*100)
    return mape_list

model.eval()
all_pred, all_true = [], []
all_day_mape = [[] for _ in range(7)]        # 收集跨场站的 Day1~7 MAPE

with torch.no_grad():
    for sid, grp in df.groupby('station_ref_id'):
        if len(grp) < CFG['past_steps'] + CFG['future_steps']:
            continue

        window = grp.tail(CFG['past_steps'] + CFG['future_steps'])
        xe = sc_enc.transform(window[ENC_COLS])[:CFG['past_steps']].astype(np.float32)
        xd = sc_dec.transform(window[DEC_COLS])[CFG['past_steps']:].astype(np.float32)
        xe = torch.from_numpy(xe).unsqueeze(0).to(device)
        xd = torch.from_numpy(xd).unsqueeze(0).to(device)

        pred_scaled = model(xe, xd).cpu().numpy().flatten()
        pred = sc_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        true = window.tail(CFG['future_steps'])['load_discharge_delta'].values

        # --- 每日 MAPE ---
        day_mape = calc_day_mape(true, pred)
        day_mape_str = ' | '.join([f'D{idx+1}:{m:.2f}%' for idx, m in enumerate(day_mape)])
        print(f'{sid}  Day-wise MAPE  {day_mape_str}')

        # 汇总
        for i, m in enumerate(day_mape):
            all_day_mape[i].append(m)
        all_pred.append(pred); all_true.append(true)

# ---- 汇总整体 ----
if all_true:
    overall_mape = mean_absolute_percentage_error(np.concatenate(all_true),
                                                  np.concatenate(all_pred))*100
    overall_rmse = np.sqrt(mean_squared_error(np.concatenate(all_true),
                                              np.concatenate(all_pred)))
    print('\nOverall Day-wise MAPE')
    for i, lst in enumerate(all_day_mape):
        print(f'  Day {i+1}: {np.mean(lst):.2f}%')

    print(f'\nOverall 7-day MAPE={overall_mape:.2f}%  RMSE={overall_rmse:.2f}')

print('\n基础模型训练 & 评估完成 ✅')