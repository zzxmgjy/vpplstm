# =========================================================
#  Encoder–Decoder  +  WeightedL1 + prev_load (Teacher-Forcing)
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
# 1️⃣ 读取
# =========================================================
df = pd.read_csv('loaddata.csv', parse_dates=['energy_date'])
df = df.sort_values(['station_ref_id', 'energy_date'])
df['code'] = df['code'].fillna(999).astype(int)

# one-hot
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

    # 天气物理
    d['dew_point']  = d['temp'] - (100-d['humidity'])/5
    d['feels_like'] = d['temp'] + 0.33*d['humidity'] - 4
    for k in [1,24]:
        d[f'temp_diff{k}'] = d['temp'].diff(k)

    # 周期
    d['sin_hour'] = np.sin(2*np.pi*(d['hour']+d['minute']/60)/24)
    d['cos_hour'] = np.cos(2*np.pi*(d['hour']+d['minute']/60)/24)
    d['sin_wday'] = np.sin(2*np.pi*d['weekday']/7)
    d['cos_wday'] = np.cos(2*np.pi*d['weekday']/7)

    # 日历
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

# prev_load = t-1 负荷，用于 decoder (未来步用 teacher forcing)
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
# 5️⃣ 缺失值处理
# =========================================================
df = df.fillna(method='ffill').fillna(method='bfill')
df = df.dropna(subset=['load_discharge_delta'])
for col in ['load_trend','load_seasonal','load_resid',
            'load_q10_48','load_q90_48','load_norm_48','prev_load']:
    if col in df.columns:
        df[col] = df[col].fillna(0)

print(f"🟢  数据行数 {len(df):,}")

# =========================================================
# 6️⃣ 特征全集
# =========================================================
ENC_FULL = [c for c in df.columns
            if c not in ['energy_date','station_ref_id','load_discharge_delta']]
DEC_FULL = [c for c in ENC_FULL if (not c.startswith('load_'))] + ['prev_load']  # 确保在 decoder

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

print(f"✅ 选 {len(ENC_COLS)} encoder  {len(DEC_COLS)} decoder 特征")

# =========================================================
# 8️⃣ 滑窗数据
# =========================================================
def make_ds(data,past,fut):
    Xp,Xf,Y=[],[],[]
    sc_e,sc_d,sc_y=StandardScaler(),StandardScaler(),StandardScaler()
    e_all=sc_e.fit_transform(data[ENC_COLS])
    d_all=sc_d.fit_transform(data[DEC_COLS])
    y_all=sc_y.fit_transform(data[['load_discharge_delta']])
    e_df=pd.DataFrame(e_all,columns=ENC_COLS,index=data.index); e_df['y']=y_all
    d_df=pd.DataFrame(d_all,columns=DEC_COLS,index=data.index)
    for sid,g in data.groupby('station_ref_id'):
        if len(g)<past+fut: continue
        e_arr=e_df.loc[g.index,ENC_COLS].values
        d_arr=d_df.loc[g.index,DEC_COLS].values
        y_arr=e_df.loc[g.index,'y'].values
        for i in range(len(g)-past-fut+1):
            Xp.append(e_arr[i:i+past])
            Xf.append(d_arr[i+past:i+past+fut])
            Y.append(y_arr[i+past:i+past+fut])
    return np.array(Xp,np.float32),np.array(Xf,np.float32),np.array(Y,np.float32),sc_e,sc_d,sc_y

Xp,Xf,Y,sc_e,sc_d,sc_y=make_ds(df,CFG['past_steps'],CFG['future_steps'])
print("🍱 样本:",len(Xp))
spl=int(.8*len(Xp))
tr_ds=TensorDataset(torch.from_numpy(Xp[:spl]),torch.from_numpy(Xf[:spl]),torch.from_numpy(Y[:spl]))
va_ds=TensorDataset(torch.from_numpy(Xp[spl:]),torch.from_numpy(Xf[spl:]),torch.from_numpy(Y[spl:]))
tr_loader=DataLoader(tr_ds,batch_size=CFG['batch_size'],shuffle=True)
va_loader=DataLoader(va_ds,batch_size=CFG['batch_size'],shuffle=False)

# =========================================================
# 9️⃣ 网络 + WeightedL1
# =========================================================
class EncDec(nn.Module):
    def __init__(self,d_enc,d_dec,hid,drop):
        super().__init__()
        self.enc=nn.LSTM(d_enc,hid,batch_first=True)
        self.dec=nn.LSTM(d_dec,hid,batch_first=True)
        self.dp=nn.Dropout(drop)
        self.fc=nn.Linear(hid,1)
    def forward(self,xe,xd):
        _,(h,c)=self.enc(xe)
        out,_=self.dec(xd,(h,c))
        return self.fc(self.dp(out)).squeeze(-1)

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
    for xe,xd,yy in tr_loader:
        xe,xd,yy=xe.to(dev),xd.to(dev),yy.to(dev)
        opt.zero_grad(); loss=crit(model(xe,xd),yy); loss.backward(); opt.step()
        tr+=loss.item()
    tr/=len(tr_loader)
    model.eval(); va=0
    with torch.no_grad():
        for xe,xd,yy in va_loader:
            xe,xd,yy=xe.to(dev),xd.to(dev),yy.to(dev)
            va+=crit(model(xe,xd),yy).item()
    va/=len(va_loader); sch.step(va)
    print(f'E{ep:03d} tr{tr:.4f} va{va:.4f}')
    if va<best:
        best=va;wait=0;torch.save(model.state_dict(),'output_pytorch/model_weighted.pth')
    else:
        wait+=1
        if wait>=CFG['patience']: print("EarlyStop");break

# ---------- 保存 ----------
joblib.dump(sc_e,'output_pytorch/scaler_enc.pkl')
joblib.dump(sc_d,'output_pytorch/scaler_dec.pkl')
joblib.dump(sc_y,'output_pytorch/scaler_y.pkl')
joblib.dump(ENC_COLS,'output_pytorch/enc_cols.pkl')
joblib.dump(DEC_COLS,'output_pytorch/dec_cols.pkl')

# =========================================================
# 🔟 评估
# =========================================================
def day_mape(t,p):
    r=[]
    for d in range(7):
        s,e=d*96,(d+1)*96
        t0,t1=t[s:e],p[s:e]
        t0=np.where(t0==0,1e-6,t0)
        r.append(mean_absolute_percentage_error(t0,t1)*100)
    return r

model.eval(); all_p,all_t=[],[]
day_res=[[] for _ in range(7)]
with torch.no_grad():
    for sid,grp in df.groupby('station_ref_id'):
        if len(grp)<CFG['past_steps']+CFG['future_steps']:continue
        win=grp.tail(CFG['past_steps']+CFG['future_steps'])
        xe=sc_e.transform(win[ENC_COLS])[:CFG['past_steps']].astype(np.float32)
        xd=sc_d.transform(win[DEC_COLS])[CFG['past_steps']:].astype(np.float32)
        xe=torch.from_numpy(xe).unsqueeze(0).to(dev)
        xd=torch.from_numpy(xd).unsqueeze(0).to(dev)
        pred_s=model(xe,xd).cpu().numpy().flatten()
        pred=sc_y.inverse_transform(pred_s.reshape(-1,1)).flatten()
        true=win.tail(CFG['future_steps'])['load_discharge_delta'].values
        dm=day_mape(true,pred)
        print(sid,["%.2f%%"%m for m in dm])
        for i,m in enumerate(dm): day_res[i].append(m)
        all_p.append(pred); all_t.append(true)
if all_t:
    overall=mean_absolute_percentage_error(np.concatenate(all_t),
                                           np.concatenate(all_p))*100
    print("\nOverall 7-day MAPE %.2f%%"%overall)
    for i,l in enumerate(day_res):print(f'Day{i+1}:{np.mean(l):.2f}%')
print(f'🏁 done!  total {time.time()-t0:.1f}s')