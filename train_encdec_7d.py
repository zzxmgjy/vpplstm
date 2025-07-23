# =========================================================
#  Encoder–Decoder  +  WeightedL1 + prev_load (Teacher-Forcing)
#  Updated for vpp_meter.csv with new field names
# =========================================================
import os, warnings, gc, time
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from statsmodels.tsa.seasonal import STL
import holidays, lightgbm as lgb
from packaging import version
import joblib

warnings.filterwarnings("ignore")
os.makedirs("output_pytorch", exist_ok=True)

# ---------- ⚙ 配置 ----------
CFG = dict(
    past_steps   = 96*5,
    future_steps = 96*7,          # 672
    hidden_dim   = 128,
    drop_rate    = .24,
    batch_size   = 128,
    epochs       = 200,
    patience     = 20,
    lr           = 2e-4,
    top_k        = 80,
    lgb_rounds   = 400,
    use_stl      = False
)

t0 = time.time()

# =========================================================
# 1️⃣ 读取 vpp_meter.csv
# =========================================================
df = pd.read_csv('vpp_meter.csv', parse_dates=['ts'])
df = df.sort_values(['station_ref_id', 'ts'])

# 检查必需字段
required_fields = ['ts', 'total_active_power', 'forward_total_active_energy', 
                   'backward_total_active_energy', 'label', 'station_ref_id']
missing_fields = [f for f in required_fields if f not in df.columns]
if missing_fields:
    raise ValueError(f"缺少必需字段: {missing_fields}")

# 特别检查功率字段，如果缺失则创建默认值
if 'total_active_power' not in df.columns:
    print("WARNING: total_active_power field missing, using default conversion from forward_total_active_energy")
    df['total_active_power'] = df['forward_total_active_energy'] * 4  # 假设15分钟电量转换为功率

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
        print(f"WARNING: Field '{field}' missing, set default value: {default_value}")

df['code'] = df['code'].fillna(999).astype(int)

# one-hot encoding for code
df = pd.concat([df, pd.get_dummies(df['code'].astype(str), prefix='code')], axis=1)

# =========================================================
# 2️⃣ 基础时间/天气/节假日特征
# =========================================================
cn_holidays = holidays.country_holidays('CN')

def make_is_peak(ts):
    h, mi = ts.dt.hour, ts.dt.minute
    return ((h>8)|((h==8)&(mi>=30))) & ((h<17)|((h==17)&(mi<=30)))

def enrich(d:pd.DataFrame):
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

    # 周期特征
    d['sin_hour'] = np.sin(2*np.pi*(d['hour']+d['minute']/60)/24)
    d['cos_hour'] = np.cos(2*np.pi*(d['hour']+d['minute']/60)/24)
    d['sin_wday'] = np.sin(2*np.pi*d['weekday']/7)
    d['cos_wday'] = np.cos(2*np.pi*d['weekday']/7)

    # 日历特征
    d['is_holiday'] = d['energy_date'].isin(cn_holidays).astype(int)
    d['is_work']    = ((d['weekday']<5)&(~d['energy_date'].isin(cn_holidays))).astype(int)
    d['is_peak']    = make_is_peak(d['energy_date']).astype(int)
    
    for lag in [1,2,3]:
        d[f'before_holiday_{lag}'] = d['energy_date'].shift(-lag).isin(cn_holidays).astype(int)
        d[f'after_holiday_{lag}']  = d['energy_date'].shift(lag ).isin(cn_holidays).astype(int)
    
    d['is_month_begin'] = (d['day']<=3).astype(int)
    d['is_month_end']   = d['energy_date'].dt.is_month_end.astype(int)
    return d

df = enrich(df)

# =========================================================
# 3️⃣ 负荷衍生 + prev_load (Teacher Forcing)
# =========================================================
for lag in [1,2,4,8,12,24,48,96]:
    df[f'load_lag{lag}'] = df.groupby('station_ref_id')['load_discharge_delta'].shift(lag)

for w in [4,8,12,24,48,96]:
    g = df.groupby('station_ref_id')['load_discharge_delta']
    df[f'load_ma{w}']  = g.rolling(w,1).mean().reset_index(level=0, drop=True)
    df[f'load_std{w}'] = g.rolling(w,1).std().reset_index(level=0, drop=True)

# prev_load = t-1 负荷，用于 decoder (未来步用 teacher forcing)
df['prev_load'] = df.groupby('station_ref_id')['load_discharge_delta'].shift(1)

# =========================================================
# 4️⃣ STL + 分位归一
# =========================================================
def add_adv(g):
    if CFG['use_stl'] and len(g)>=96*14:
        res = STL(g['load_discharge_delta'], period=96*7, robust=True).fit()
        g['load_trend']    = res.trend
        g['load_seasonal'] = res.seasonal
        g['load_resid']    = res.resid
    else:
        g[['load_trend','load_seasonal','load_resid']] = np.nan
    g['load_q10_48'] = g['load_discharge_delta'].rolling(48,1).quantile(.1)
    g['load_q90_48'] = g['load_discharge_delta'].rolling(48,1).quantile(.9)
    g['load_norm_48']= g['load_discharge_delta'] / g['load_q90_48']
    return g

df = df.groupby('station_ref_id', group_keys=False).apply(add_adv)

# =========================================================
# 5️⃣ 缺失值处理 + 数据清洗
# =========================================================
df = df.fillna(method='ffill').fillna(method='bfill')
df = df.dropna(subset=['load_discharge_delta'])
for col in ['load_trend','load_seasonal','load_resid',
            'load_q10_48','load_q90_48','load_norm_48','prev_load']:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# 清理无穷大和异常值
print("🧹 清理数据中的无穷大和异常值...")
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col in df.columns:
        # 替换无穷大值
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        # 填充 NaN
        df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)
        # 处理极端异常值 (超过99.9%分位数的值)
        if df[col].std() > 0:
            upper_bound = df[col].quantile(0.999)
            lower_bound = df[col].quantile(0.001)
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

print(f"🟢  数据行数 {len(df):,}")

# =========================================================
# 6️⃣ 特征全集
# =========================================================
ENC_FULL = [c for c in df.columns
            if c not in ['energy_date','station_ref_id','load_discharge_delta',
                        'total_active_power','backward_total_active_energy','label']]
DEC_FULL = [c for c in ENC_FULL if (not c.startswith('load_'))] + ['prev_load']

# =========================================================
# 7️⃣ LightGBM 选 Top-K
# =========================================================
rows = len(df)
sample_df = df.sample(frac=0.2, random_state=42) if rows>=5000 else df.copy()
X, y = sample_df[ENC_FULL], sample_df['load_discharge_delta']
split = max(int(.8*len(sample_df)),1)
split = len(sample_df)-1 if split==len(sample_df) else split
Xtr,Xva,ytr,yva = X.iloc[:split],X.iloc[split:],y.iloc[:split],y.iloc[split:]

train_ds = lgb.Dataset(Xtr,ytr,categorical_feature=['code'] if 'code' in X.columns else [])
val_ds   = lgb.Dataset(Xva,yva,reference=train_ds,
                       categorical_feature=['code'] if 'code' in X.columns else [])

params=dict(objective='regression',metric='l1',learning_rate=.05,
            num_leaves=31,feature_fraction=.8,bagging_fraction=.8,
            bagging_freq=5,verbose=-1)
ver=version.parse(lgb.__version__)
if ver<version.parse("4.0"):
    gbm=lgb.train(params,train_ds,CFG['lgb_rounds'],
                  valid_sets=[train_ds,val_ds],
                  early_stopping_rounds=50,verbose_eval=False)
else:
    gbm=lgb.train(params,train_ds,CFG['lgb_rounds'],
                  valid_sets=[train_ds,val_ds],
                  callbacks=[lgb.early_stopping(50,verbose=False)])

imp=pd.DataFrame({'feat':ENC_FULL,
                  'gain':gbm.feature_importance('gain')}).sort_values('gain',ascending=False)
ENC_COLS=imp.head(CFG['top_k'])['feat'].tolist()
if 'prev_load' not in ENC_COLS: ENC_COLS.append('prev_load')
DEC_COLS=[c for c in ENC_COLS if not c.startswith('load_')]
if 'prev_load' not in DEC_COLS: DEC_COLS.append('prev_load')

print(f"SUCCESS: Selected {len(ENC_COLS)} encoder  {len(DEC_COLS)} decoder features")

# =========================================================
# 8️⃣ 滑窗数据 - 双输出（电量+功率）
# =========================================================
def make_ds(data,past,fut):
    Xp,Xf,Y_energy,Y_power=[],[],[],[]
    sc_e,sc_d,sc_y_energy,sc_y_power=StandardScaler(),StandardScaler(),StandardScaler(),StandardScaler()
    
    # 最终数据验证
    print("INFO: Validating data quality...")
    for col in ENC_COLS + DEC_COLS + ['load_discharge_delta', 'total_active_power']:
        if col in data.columns:
            if data[col].isnull().any():
                print(f"WARNING: {col} has {data[col].isnull().sum()} null values, filling with median")
                data[col] = data[col].fillna(data[col].median())
            if np.isinf(data[col]).any():
                print(f"WARNING: {col} has infinite values, replacing with boundary values")
                data[col] = data[col].replace([np.inf, -np.inf], [data[col].quantile(0.99), data[col].quantile(0.01)])
    
    e_all=sc_e.fit_transform(data[ENC_COLS])
    d_all=sc_d.fit_transform(data[DEC_COLS])
    y_energy_all=sc_y_energy.fit_transform(data[['load_discharge_delta']])
    y_power_all=sc_y_power.fit_transform(data[['total_active_power']])
    e_df=pd.DataFrame(e_all,columns=ENC_COLS,index=data.index)
    e_df['y_energy']=y_energy_all
    e_df['y_power']=y_power_all
    d_df=pd.DataFrame(d_all,columns=DEC_COLS,index=data.index)
    for sid,g in data.groupby('station_ref_id'):
        if len(g)<past+fut: continue
        e_arr=e_df.loc[g.index,ENC_COLS].values
        d_arr=d_df.loc[g.index,DEC_COLS].values
        y_energy_arr=e_df.loc[g.index,'y_energy'].values
        y_power_arr=e_df.loc[g.index,'y_power'].values
        for i in range(len(g)-past-fut+1):
            Xp.append(e_arr[i:i+past])
            Xf.append(d_arr[i+past:i+past+fut])
            Y_energy.append(y_energy_arr[i+past:i+past+fut])
            Y_power.append(y_power_arr[i+past:i+past+fut])
    return (np.array(Xp,np.float32),np.array(Xf,np.float32),
            np.array(Y_energy,np.float32),np.array(Y_power,np.float32),
            sc_e,sc_d,sc_y_energy,sc_y_power)

Xp,Xf,Y_energy,Y_power,sc_e,sc_d,sc_y_energy,sc_y_power=make_ds(df,CFG['past_steps'],CFG['future_steps'])
print("🍱 样本:",len(Xp))
spl=int(.8*len(Xp))
tr_ds=TensorDataset(torch.from_numpy(Xp[:spl]),torch.from_numpy(Xf[:spl]),
                    torch.from_numpy(Y_energy[:spl]),torch.from_numpy(Y_power[:spl]))
va_ds=TensorDataset(torch.from_numpy(Xp[spl:]),torch.from_numpy(Xf[spl:]),
                    torch.from_numpy(Y_energy[spl:]),torch.from_numpy(Y_power[spl:]))
tr_loader=DataLoader(tr_ds,batch_size=CFG['batch_size'],shuffle=True)
va_loader=DataLoader(va_ds,batch_size=CFG['batch_size'],shuffle=False)

# =========================================================
# 9️⃣ 网络 + WeightedL1 - 双输出
# =========================================================
class EncDec(nn.Module):
    def __init__(self,d_enc,d_dec,hid,drop):
        super().__init__()
        self.enc=nn.LSTM(d_enc,hid,batch_first=True)
        self.dec=nn.LSTM(d_dec,hid,batch_first=True)
        self.dp=nn.Dropout(drop)
        self.fc_energy=nn.Linear(hid,1)  # 电量预测
        self.fc_power=nn.Linear(hid,1)   # 功率预测
    def forward(self,xe,xd):
        _,(h,c)=self.enc(xe)
        out,_=self.dec(xd,(h,c))
        out_dp = self.dp(out)
        energy_pred = self.fc_energy(out_dp).squeeze(-1)
        power_pred = self.fc_power(out_dp).squeeze(-1)
        return energy_pred, power_pred

# 加权 L1 —— Day3,4 权重大
class WeightedL1(nn.Module):
    def __init__(self,fut,device):
        super().__init__()
        w=np.concatenate([
            np.ones(96*2),           # Day1-2
            np.ones(96)*1.3,         # Day3
            np.ones(96)*1.5,         # Day4
            np.ones(96*3)*1.2        # Day5-7
        ])
        self.register_buffer('w',torch.tensor(w,dtype=torch.float32,device=device))
    def forward(self,pred,target):
        return torch.mean(self.w*torch.abs(pred-target))

dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=EncDec(len(ENC_COLS),len(DEC_COLS),CFG['hidden_dim'],CFG['drop_rate']).to(dev)
crit = WeightedL1(CFG['future_steps'],dev)
opt  = torch.optim.Adam(model.parameters(),lr=CFG['lr'])
sch  = ReduceLROnPlateau(opt,'min',patience=5,factor=.5)

best=1e9;wait=0
print("⏳ 训练...")
for ep in range(1,CFG['epochs']+1):
    model.train(); tr=0
    for xe,xd,yy_energy,yy_power in tr_loader:
        xe,xd,yy_energy,yy_power=xe.to(dev),xd.to(dev),yy_energy.to(dev),yy_power.to(dev)
        opt.zero_grad()
        pred_energy, pred_power = model(xe,xd)
        loss_energy = crit(pred_energy, yy_energy)
        loss_power = crit(pred_power, yy_power)
        loss = loss_energy + loss_power  # 总损失
        loss.backward(); opt.step()
        tr+=loss.item()
    tr/=len(tr_loader)
    model.eval(); va=0
    with torch.no_grad():
        for xe,xd,yy_energy,yy_power in va_loader:
            xe,xd,yy_energy,yy_power=xe.to(dev),xd.to(dev),yy_energy.to(dev),yy_power.to(dev)
            pred_energy, pred_power = model(xe,xd)
            loss_energy = crit(pred_energy, yy_energy)
            loss_power = crit(pred_power, yy_power)
            va+=(loss_energy + loss_power).item()
    va/=len(va_loader); sch.step(va)
    print(f'E{ep:03d} tr{tr:.4f} va{va:.4f}')
    if va<best:
        best=va;wait=0;torch.save(model.state_dict(),'output_pytorch/model_weighted.pth')
    else:
        wait+=1
        if wait>=CFG['patience']: print("INFO: EarlyStop");break

# ---------- 保存 ----------
joblib.dump(sc_e,'output_pytorch/scaler_enc.pkl')
joblib.dump(sc_d,'output_pytorch/scaler_dec.pkl')
joblib.dump(sc_y_energy,'output_pytorch/scaler_y_energy.pkl')
joblib.dump(sc_y_power,'output_pytorch/scaler_y_power.pkl')
joblib.dump(ENC_COLS,'output_pytorch/enc_cols.pkl')
joblib.dump(DEC_COLS,'output_pytorch/dec_cols.pkl')

# =========================================================
# 🔟 评估 - 双输出
# =========================================================
def day_mape(t,p):
    r=[]
    for d in range(7):
        s,e=d*96,(d+1)*96
        t0,t1=t[s:e],p[s:e]
        # 更严格的处理：过滤掉异常值
        mask = (np.abs(t0) > 1e-3) & np.isfinite(t0) & np.isfinite(t1)
        if mask.sum() == 0:
            r.append(0.0)  # 如果没有有效数据，返回0
        else:
            t0_filtered = t0[mask]
            t1_filtered = t1[mask]
            # 使用绝对值确保分母为正
            t0_filtered = np.where(np.abs(t0_filtered) < 1e-3, 
                                 np.sign(t0_filtered) * 1e-3, t0_filtered)
            mape = np.mean(np.abs((t0_filtered - t1_filtered) / t0_filtered)) * 100
            # 限制MAPE的最大值，避免极端情况
            r.append(min(mape, 1000.0))
    return r

model.eval()
all_p_energy,all_t_energy=[],[]
all_p_power,all_t_power=[],[]
day_res_energy=[[] for _ in range(7)]
day_res_power=[[] for _ in range(7)]

with torch.no_grad():
    for sid,grp in df.groupby('station_ref_id'):
        if len(grp)<CFG['past_steps']+CFG['future_steps']:continue
        win=grp.tail(CFG['past_steps']+CFG['future_steps'])
        xe=sc_e.transform(win[ENC_COLS])[:CFG['past_steps']].astype(np.float32)
        xd=sc_d.transform(win[DEC_COLS])[CFG['past_steps']:].astype(np.float32)
        xe=torch.from_numpy(xe).unsqueeze(0).to(dev)
        xd=torch.from_numpy(xd).unsqueeze(0).to(dev)
        
        pred_energy_s, pred_power_s = model(xe,xd)
        pred_energy_s = pred_energy_s.cpu().numpy().flatten()
        pred_power_s = pred_power_s.cpu().numpy().flatten()
        
        pred_energy=sc_y_energy.inverse_transform(pred_energy_s.reshape(-1,1)).flatten()
        pred_power=sc_y_power.inverse_transform(pred_power_s.reshape(-1,1)).flatten()
        
        true_energy=win.tail(CFG['future_steps'])['load_discharge_delta'].values
        true_power=win.tail(CFG['future_steps'])['total_active_power'].values
        
        # 数据质量检查
        if sid == list(df['station_ref_id'].unique())[0]:  # 只对第一个站点打印调试信息
            print(f"DEBUG: Station {sid}:")
            print(f"  True energy range: {true_energy.min():.2f} ~ {true_energy.max():.2f}")
            print(f"  Pred energy range: {pred_energy.min():.2f} ~ {pred_energy.max():.2f}")
            print(f"  True power range: {true_power.min():.2f} ~ {true_power.max():.2f}")
            print(f"  Pred power range: {pred_power.min():.2f} ~ {pred_power.max():.2f}")
            print(f"  Energy zero/negative count: {(true_energy <= 0).sum()}")
            print(f"  Power zero/negative count: {(true_power <= 0).sum()}")
        
        dm_energy=day_mape(true_energy,pred_energy)
        dm_power=day_mape(true_power,pred_power)
        
        print(f"{sid} Energy: {['%.2f%%'%m for m in dm_energy]}")
        print(f"{sid} Power:  {['%.2f%%'%m for m in dm_power]}")
        
        for i,m in enumerate(dm_energy): day_res_energy[i].append(m)
        for i,m in enumerate(dm_power): day_res_power[i].append(m)
        
        all_p_energy.append(pred_energy); all_t_energy.append(true_energy)
        all_p_power.append(pred_power); all_t_power.append(true_power)

if all_t_energy:
    # 安全的MAPE计算
    def safe_mape(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mask = (np.abs(y_true) > 1e-3) & np.isfinite(y_true) & np.isfinite(y_pred)
        if mask.sum() == 0:
            return 0.0
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        y_true_filtered = np.where(np.abs(y_true_filtered) < 1e-3, 
                                 np.sign(y_true_filtered) * 1e-3, y_true_filtered)
        mape = np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100
        return min(mape, 1000.0)
    
    overall_energy = safe_mape(np.concatenate(all_t_energy), np.concatenate(all_p_energy))
    overall_power = safe_mape(np.concatenate(all_t_power), np.concatenate(all_p_power))
    print(f"\nOverall 7-day Energy MAPE: {overall_energy:.2f}%")
    print(f"Overall 7-day Power MAPE:  {overall_power:.2f}%")
    
    print("\nEnergy Day-wise MAPE:")
    for i,l in enumerate(day_res_energy):print(f'Day{i+1}:{np.mean(l):.2f}%')
    print("\nPower Day-wise MAPE:")
    for i,l in enumerate(day_res_power):print(f'Day{i+1}:{np.mean(l):.2f}%')

print(f'SUCCESS: Training completed! Total time: {time.time()-t0:.1f}s')
