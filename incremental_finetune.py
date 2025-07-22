# incremental_finetune.py
# =========================================================
# 增量微调单站模型  (与 WeightedL1 + prev_load 逻辑一致)
# =========================================================
import os, warnings, joblib, holidays, pandas as pd, numpy as np
from sklearn.metrics import mean_absolute_percentage_error
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
STATION_ID  = 3205103743359      # ← 目标场站
NEW_CSV     = 'loaddata.csv'     # ← 新增数据 (近 15 天 或任意周期)
EPOCHS      = 15
BATCH_SIZE  = 128
LR_INCR     = 2e-4               # 增量学习率
# ---------------------------------------

ROOT        = 'output_pytorch'
MODEL_DIR   = f'{ROOT}/model_{STATION_ID}'
os.makedirs(MODEL_DIR, exist_ok=True)

PAST, FUT   = 96*5, 96*7
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- is_peak ----------
def make_is_peak(ts):
    h,mi = ts.dt.hour, ts.dt.minute
    return (((h>8)|((h==8)&(mi>=30))) & ((h<17)|((h==17)&(mi<=30)))).astype(int)

# ---------- 读取尾巴缓存 (上次 PAST+FUT) ----------
TAIL_PATH = f'{MODEL_DIR}/tail_cache_{STATION_ID}.csv'
old_tail  = pd.read_csv(TAIL_PATH, parse_dates=['energy_date']) if os.path.exists(TAIL_PATH) else pd.DataFrame()

# ---------- 读取新增 CSV ----------
df_new = pd.read_csv(NEW_CSV, parse_dates=['energy_date'])
if 'station_ref_id' in df_new.columns:
    df_new = df_new[df_new['station_ref_id'] == STATION_ID]
if df_new.empty:
    raise ValueError('新增 CSV 中无该场站数据')

# 合并、按时间排序
df = pd.concat([old_tail, df_new], ignore_index=True).sort_values('energy_date').reset_index(drop=True)

# ---------- enrich (必须与总模型一致) ----------
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

# ---------- 滞后 / rolling + prev_load ----------
for lag in [1,2,4,8,12,24,48,96]:
    df[f'load_lag{lag}'] = df['load_discharge_delta'].shift(lag)
for w in [4,8,12,24,48,96]:
    df[f'load_ma{w}']  = df['load_discharge_delta'].rolling(w,1).mean()
    df[f'load_std{w}'] = df['load_discharge_delta'].rolling(w,1).std()
df['prev_load'] = df['load_discharge_delta'].shift(1)

df = df.fillna(method='ffill').fillna(method='bfill').dropna()

# ---------- 载入特征列 & Scaler ----------
enc_cols = joblib.load(f'{ROOT}/enc_cols.pkl')
dec_cols = joblib.load(f'{ROOT}/dec_cols.pkl')
sc_enc   = joblib.load(f'{ROOT}/scaler_enc.pkl')
sc_dec   = joblib.load(f'{ROOT}/scaler_dec.pkl')
sc_y     = joblib.load(f'{ROOT}/scaler_y.pkl')

# 补全缺失列
for col in set(enc_cols + dec_cols):
    if col not in df.columns:
        df[col] = 0

# ---------- 滑窗 ----------
def make_ds(data):
    Xp,Xf,Y = [],[],[]
    enc = sc_enc.transform(data[enc_cols])
    dec = sc_dec.transform(data[dec_cols])
    y   = sc_y.transform(data[['load_discharge_delta']])
    for i in range(len(data)-PAST-FUT+1):
        Xp.append(enc[i:i+PAST])
        Xf.append(dec[i+PAST:i+PAST+FUT])
        Y.append(y[i+PAST:i+PAST+FUT,0])
    return np.array(Xp,np.float32), np.array(Xf,np.float32), np.array(Y,np.float32)

Xp,Xf,Y = make_ds(df)
if len(Xp)==0:
    raise ValueError('增量数据不足以形成滑窗样本')
loader = DataLoader(TensorDataset(torch.from_numpy(Xp),
                                  torch.from_numpy(Xf),
                                  torch.from_numpy(Y)),
                    batch_size=BATCH_SIZE, shuffle=True)

# ---------- 模型 ----------
class EncDec(nn.Module):
    def __init__(self,d_enc,d_dec,hid,drop):
        super().__init__()
        self.enc = nn.LSTM(d_enc,hid,batch_first=True)
        self.dec = nn.LSTM(d_dec,hid,batch_first=True)
        self.dp  = nn.Dropout(drop); self.fc = nn.Linear(hid,1)
    def forward(self,xe,xd):
        _,(h,c)=self.enc(xe)
        out,_ = self.dec(xd,(h,c))
        return self.fc(self.dp(out)).squeeze(-1)

# --- Weighted L1 (与总模型保持一致) ---
class WeightedL1(nn.Module):
    def __init__(self,fut,device):
        super().__init__()
        w = np.concatenate([
            np.ones(96*2),
            np.ones(96)*1.3,
            np.ones(96)*1.5,
            np.ones(96*3)*1.2])
        self.register_buffer('w', torch.tensor(w,dtype=torch.float32,device=device))
    def forward(self,pred,tgt):
        return torch.mean(self.w*torch.abs(pred-tgt))

model = EncDec(len(enc_cols), len(dec_cols), 128, .24).to(DEVICE)

# --- 载入历史权重 ---
opt_path = f'{MODEL_DIR}/model_optimized_{STATION_ID}.pth'
base_path= f'{ROOT}/model_weighted.pth'   # 你的总模型权重
if os.path.exists(opt_path):
    model.load_state_dict(torch.load(opt_path, map_location=DEVICE))
    print(">>> 加载已微调权重")
else:
    model.load_state_dict(torch.load(base_path, map_location=DEVICE))
    print(">>> 加载基础权重（首次增量）")

# 冻结 Encoder
for p in model.enc.parameters():
    p.requires_grad_(False)

criterion = WeightedL1(FUT, DEVICE)
optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=LR_INCR)
scheduler = ReduceLROnPlateau(optimizer,'min',patience=2,factor=.5)

best=float('inf'); wait=0
print(f"🚀 增量训练样本 {len(loader.dataset)}   batch {BATCH_SIZE}")
for ep in range(1,EPOCHS+1):
    model.train(); ep_loss=0
    for xe,xd,yy in loader:
        xe,xd,yy = xe.to(DEVICE),xd.to(DEVICE),yy.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xe,xd),yy)
        loss.backward(); optimizer.step()
        ep_loss += loss.item()
    ep_loss /= len(loader); scheduler.step(ep_loss)

    log=f'E{ep:02d} loss {ep_loss:.4f}'
    if ep_loss<best:
        best=ep_loss; wait=0
        torch.save(model.state_dict(), opt_path)
        log+=' ✔ save'
    else:
        wait+=1
        if wait>=4:
            log+=' (early stop)'; print(log); break
    print(log)

# ---------- 简易评估 ----------
def day_mape(t,p):
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

    overall = mean_absolute_percentage_error(
        np.where(true==0,1e-6,true), pred)*100
    dm = day_mape(true,pred)
print(f'\n📊 7-day MAPE {overall:.2f}%  | '+
      '  '.join([f'D{i+1}:{m:.2f}%' for i,m in enumerate(dm)]))

# ---------- 更新尾巴缓存 ----------
df.tail(PAST+FUT).to_csv(TAIL_PATH,index=False)
print("\n✅ 增量微调完成，权重 & 尾巴缓存已更新")