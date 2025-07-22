# incremental_finetune.py
# =========================================================
# 增量微调脚本（每 15 天或任意周期）
# 修改 CONFIG 部分即可运行
# =========================================================
import os, warnings, holidays, joblib, pandas as pd, numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
STATION_ID  = 1749985246419357696                    # 目标场站
NEW_CSV     = 'load.csv'  # 新增 15 天数据
EPOCHS      = 15
BATCH_SIZE  = 128
LR_INCR     = 2.16485e-4                      # 增量更小 LR
# ----------------------------------------

ROOT        = 'output_pytorch'
MODE_DIR    = os.path.join(ROOT, f'mode_{STATION_ID}')
os.makedirs(MODE_DIR, exist_ok=True)

PAST, FUT   = 96*5, 96*7                 # 480 / 672
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- PEAK 时段 08:30~17:30 ----------
def make_is_peak(ts):
    h  = ts.dt.hour; mi = ts.dt.minute
    return (((h>8)|((h==8)&(mi>=30))) & ((h<17)|((h==17)&(mi<=30)))).astype(int)

# ---------- 读取尾巴缓存 ----------
TAIL_PATH = os.path.join(MODE_DIR, f'tail_cache_{STATION_ID}.csv')
if os.path.exists(TAIL_PATH):
    old_tail = pd.read_csv(TAIL_PATH, parse_dates=['energy_date'])
else:
    old_tail = pd.DataFrame()

# ---------- 读取新增 CSV ----------
df_new = pd.read_csv(NEW_CSV, parse_dates=['energy_date'])
if 'station_ref_id' in df_new.columns:
    df_new = df_new[df_new['station_ref_id'] == STATION_ID]
if df_new.empty: raise ValueError('新增 CSV 中无该场站数据')

df = pd.concat([old_tail, df_new], ignore_index=True)
df = df.sort_values('energy_date').reset_index(drop=True)

# ---------- enrich 函数 ----------
CN_HOLIDAYS = holidays.country_holidays('CN')
def enrich(d):
    d['hour']     = d['energy_date'].dt.hour
    d['minute']   = d['energy_date'].dt.minute
    d['weekday']  = d['energy_date'].dt.weekday
    d['temp_squared'] = d['temp']**2
    d['sin_hour'] = np.sin(2*np.pi*(d['hour']+d['minute']/60)/24)
    d['cos_hour'] = np.cos(2*np.pi*(d['hour']+d['minute']/60)/24)
    d['sin_wday'] = np.sin(2*np.pi*d['weekday']/7)
    d['cos_wday'] = np.cos(2*np.pi*d['weekday']/7)
    d['is_holiday'] = d['energy_date'].isin(CN_HOLIDAYS).astype(int)
    if 'is_work' not in d.columns:
        d['is_work'] = ((d['weekday']<5) & (d['is_holiday']==0)).astype(int)
    if 'is_peak' not in d.columns:
        d['is_peak'] = make_is_peak(d['energy_date'])
    return d
df = enrich(df)

# ---------- 负荷衍生特征 ----------
for lag in [1,4,24,96]:
    df[f'load_lag{lag}'] = df['load_discharge_delta'].shift(lag)
for w in [4,24,96]:
    df[f'load_ma{w}']  = df['load_discharge_delta'].rolling(w,1).mean()
    df[f'load_std{w}'] = df['load_discharge_delta'].rolling(w,1).std()
df = df.fillna(method='ffill').dropna()

# ---------- Feature Lists (dump / load) ----------
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
# 把列列表也存一份，方便其他脚本统一使用
joblib.dump(ENC_COLS, f'{ROOT}/enc_cols.pkl')
joblib.dump(DEC_COLS, f'{ROOT}/dec_cols.pkl')

# ---------- Scaler ----------
sc_enc = joblib.load(f'{ROOT}/scaler_enc.pkl')
sc_dec = joblib.load(f'{ROOT}/scaler_dec.pkl')
sc_y   = joblib.load(f'{ROOT}/scaler_y.pkl')

def windowize(data):
    Xp,Xf,Y = [],[],[]
    enc = sc_enc.transform(data[ENC_COLS])
    dec = sc_dec.transform(data[DEC_COLS])
    y   = sc_y.transform(data[['load_discharge_delta']])
    for i in range(len(data)-PAST-FUT+1):
        Xp.append(enc[i:i+PAST])
        Xf.append(dec[i+PAST:i+PAST+FUT])
        Y.append(y[i+PAST:i+PAST+FUT,0])
    return np.array(Xp,np.float32), np.array(Xf,np.float32), np.array(Y,np.float32)

Xp,Xf,Y = windowize(df)
if len(Xp)==0: raise ValueError('增量数据不足形成任何滑窗样本')

loader = DataLoader(TensorDataset(torch.from_numpy(Xp),
                                  torch.from_numpy(Xf),
                                  torch.from_numpy(Y)),
                    batch_size=BATCH_SIZE, shuffle=True)

# ---------- 模型 ----------
class EncDec(nn.Module):
    def __init__(self,d1,d2,hid,fut,drop):
        super().__init__()
        self.enc = nn.LSTM(d1,hid,batch_first=True)
        self.dec = nn.LSTM(d2,hid,batch_first=True)
        self.dp  = nn.Dropout(drop); self.fc=nn.Linear(hid,1)
    def forward(self,xe,xd):
        _,(h,c)=self.enc(xe)
        y,_=self.dec(xd,(h,c))
        return self.fc(self.dp(y)).squeeze(-1)

model = EncDec(len(ENC_COLS),len(DEC_COLS),128,FUT,.24).to(DEVICE)
opt_path = os.path.join(MODE_DIR, f'model_optimized_{STATION_ID}.pth')
if os.path.exists(opt_path):
    model.load_state_dict(torch.load(opt_path, map_location=DEVICE))
    print('>>> 载入既有微调权重')
else:
    model.load_state_dict(torch.load(f'{ROOT}/model_base_encdec7d.pth', map_location=DEVICE))
    print('>>> 首次微调：载入基础权重')

for p in model.enc.parameters(): p.requires_grad_(False)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_INCR)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=.5)

best = float('inf'); pat = 0
for ep in range(1, EPOCHS+1):
    model.train(); ep_loss = 0
    for xe,xd,yy in loader:
        xe,xd,yy = xe.to(DEVICE), xd.to(DEVICE), yy.to(DEVICE)
        optimizer.zero_grad(); loss = criterion(model(xe,xd), yy)
        loss.backward(); optimizer.step()
        ep_loss += loss.item()
    ep_loss /= len(loader)
    scheduler.step(ep_loss)
    msg = f'Increment E{ep:02d} loss {ep_loss:.4f}'
    if ep_loss < best:
        best = ep_loss; pat = 0
        torch.save(model.state_dict(), opt_path)
        msg += ' ✔ save'
    else:
        pat += 1
        if pat >= 4: msg += ' (early)'; print(msg); break
    print(msg)

# ---------- Quick Evaluation (整体 + 按天) ----------
def calc_day_mape(t, p):
    res=[]
    for d in range(7):
        s,e=d*96,(d+1)*96
        res.append(mean_absolute_percentage_error(
              np.where(t[s:e]==0,1e-6,t[s:e]), p[s:e])*100)
    return res

model.eval()
with torch.no_grad():
    xe=torch.from_numpy(Xp[-1:].astype(np.float32)).to(DEVICE)
    xd=torch.from_numpy(Xf[-1:].astype(np.float32)).to(DEVICE)
    pred_s=model(xe,xd).cpu().numpy().flatten()
    pred  = sc_y.inverse_transform(pred_s.reshape(-1,1)).flatten()
    true  = sc_y.inverse_transform(Y[-1:].reshape(-1,1)).flatten()

    mape_all = mean_absolute_percentage_error(
        np.where(true==0,1e-6,true), pred)*100
    day_mapes = calc_day_mape(true, pred)
    print(f'\n【{STATION_ID}】7-day MAPE {mape_all:.2f}%')
    print('Day-wise MAPE: ' + ' | '.join([f'D{d+1}:{m:.2f}%' for d,m in enumerate(day_mapes)]))

# ---------- 更新尾巴缓存 (保留 PAST+FUT 行) ----------
df.tail(PAST+FUT).to_csv(TAIL_PATH, index=False)
print('\n增量微调完成，权重 & 尾巴缓存已更新 ✅')
