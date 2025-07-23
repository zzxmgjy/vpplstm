# incremental_finetune.py
# =========================================================
# 增量微调单站模型  (与 WeightedL1 + prev_load 逻辑一致)
# Updated for vpp_meter.csv with dual outputs (energy + power)
# =========================================================
import os, warnings, joblib, holidays, pandas as pd, numpy as np
from sklearn.metrics import mean_absolute_percentage_error
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
STATION_ID  = 3205103743359      # ← 目标场站
NEW_CSV     = 'vpp_meter.csv'   # ← 新增数据使用 vpp_meter.csv
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
old_tail  = pd.read_csv(TAIL_PATH, parse_dates=['ts']) if os.path.exists(TAIL_PATH) else pd.DataFrame()
if not old_tail.empty and 'ts' in old_tail.columns:
    old_tail = old_tail.rename(columns={'ts': 'energy_date'})

# ---------- 读取新增 CSV ----------
df_new = pd.read_csv(NEW_CSV, parse_dates=['ts'])
if 'station_ref_id' in df_new.columns:
    df_new = df_new[df_new['station_ref_id'] == STATION_ID]
if df_new.empty:
    raise ValueError('新增 CSV 中无该场站数据')

# 重命名字段以保持代码兼容性
df_new = df_new.rename(columns={
    'ts': 'energy_date',
    'forward_total_active_energy': 'load_discharge_delta'
})

# 合并、按时间排序
df = pd.concat([old_tail, df_new], ignore_index=True).sort_values('energy_date').reset_index(drop=True)

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

# ---------- enrich (必须与总模型一致) ----------
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
sc_y_energy = joblib.load(f'{ROOT}/scaler_y_energy.pkl')
sc_y_power  = joblib.load(f'{ROOT}/scaler_y_power.pkl')

# 补全缺失列
for col in set(enc_cols + dec_cols):
    if col not in df.columns:
        df[col] = 0

# ---------- 滑窗 - 双输出 ----------
def make_ds(data):
    Xp,Xf,Y_energy,Y_power = [],[],[],[]
    enc = sc_enc.transform(data[enc_cols])
    dec = sc_dec.transform(data[dec_cols])
    y_energy = sc_y_energy.transform(data[['load_discharge_delta']])
    y_power  = sc_y_power.transform(data[['total_active_power']])
    for i in range(len(data)-PAST-FUT+1):
        Xp.append(enc[i:i+PAST])
        Xf.append(dec[i+PAST:i+PAST+FUT])
        Y_energy.append(y_energy[i+PAST:i+PAST+FUT,0])
        Y_power.append(y_power[i+PAST:i+PAST+FUT,0])
    return (np.array(Xp,np.float32), np.array(Xf,np.float32), 
            np.array(Y_energy,np.float32), np.array(Y_power,np.float32))

Xp,Xf,Y_energy,Y_power = make_ds(df)
if len(Xp)==0:
    raise ValueError('增量数据不足以形成滑窗样本')
loader = DataLoader(TensorDataset(torch.from_numpy(Xp),
                                  torch.from_numpy(Xf),
                                  torch.from_numpy(Y_energy),
                                  torch.from_numpy(Y_power)),
                    batch_size=BATCH_SIZE, shuffle=True)

# ---------- 模型 - 双输出 ----------
class EncDec(nn.Module):
    def __init__(self,d_enc,d_dec,hid,drop):
        super().__init__()
        self.enc = nn.LSTM(d_enc,hid,batch_first=True)
        self.dec = nn.LSTM(d_dec,hid,batch_first=True)
        self.dp  = nn.Dropout(drop)
        self.fc_energy = nn.Linear(hid,1)  # 电量预测
        self.fc_power  = nn.Linear(hid,1)  # 功率预测
    def forward(self,xe,xd):
        _,(h,c)=self.enc(xe)
        out,_ = self.dec(xd,(h,c))
        out_dp = self.dp(out)
        energy_pred = self.fc_energy(out_dp).squeeze(-1)
        power_pred = self.fc_power(out_dp).squeeze(-1)
        return energy_pred, power_pred

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
    for xe,xd,yy_energy,yy_power in loader:
        xe,xd,yy_energy,yy_power = xe.to(DEVICE),xd.to(DEVICE),yy_energy.to(DEVICE),yy_power.to(DEVICE)
        optimizer.zero_grad()
        pred_energy, pred_power = model(xe,xd)
        loss_energy = criterion(pred_energy,yy_energy)
        loss_power = criterion(pred_power,yy_power)
        loss = loss_energy + loss_power
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

# ---------- 简易评估 - 双输出 ----------
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
    pred_energy_s, pred_power_s = model(xe,xd)
    pred_energy_s = pred_energy_s.cpu().numpy().flatten()
    pred_power_s = pred_power_s.cpu().numpy().flatten()
    
    pred_energy = sc_y_energy.inverse_transform(pred_energy_s.reshape(-1,1)).flatten()
    pred_power = sc_y_power.inverse_transform(pred_power_s.reshape(-1,1)).flatten()
    
    true_energy = sc_y_energy.inverse_transform(Y_energy[-1:].reshape(-1,1)).flatten()
    true_power = sc_y_power.inverse_transform(Y_power[-1:].reshape(-1,1)).flatten()

    overall_energy = mean_absolute_percentage_error(
        np.where(true_energy==0,1e-6,true_energy), pred_energy)*100
    overall_power = mean_absolute_percentage_error(
        np.where(true_power==0,1e-6,true_power), pred_power)*100
    
    dm_energy = day_mape(true_energy,pred_energy)
    dm_power = day_mape(true_power,pred_power)

print(f'\n📊 Energy 7-day MAPE {overall_energy:.2f}%  | '+
      '  '.join([f'D{i+1}:{m:.2f}%' for i,m in enumerate(dm_energy)]))
print(f'📊 Power  7-day MAPE {overall_power:.2f}%   | '+
      '  '.join([f'D{i+1}:{m:.2f}%' for i,m in enumerate(dm_power)]))

# ---------- 更新尾巴缓存 ----------
# 保存时使用原始字段名 ts
df_cache = df.tail(PAST+FUT).copy()
df_cache = df_cache.rename(columns={'energy_date': 'ts'})
df_cache.to_csv(TAIL_PATH,index=False)
print("\n✅ 增量微调完成，权重 & 尾巴缓存已更新")