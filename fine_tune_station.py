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
STATION_ID  = 1851144626925211648       # ← 修改为目标场站
CSV_FILE    = 'vpp_meter.csv'    # ← 使用 vpp_meter.csv
EPOCHS      = 80                  # 增加训练轮数
BATCH_SIZE  = 64                  # 减小批次大小，提高训练稳定性
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
        print(f"WARNING: Field '{field}' missing, set default value: {default_value}")

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

# ---------- 增强特征工程 ----------
# 基础滞后特征
for lag in [1,2,4,8,12,24,48,96,192,288]:  # 增加更长期滞后
    df[f'load_lag{lag}'] = df['load_discharge_delta'].shift(lag)

# 滚动统计特征
for w in [4,8,12,24,48,96,192]:  # 增加更长窗口
    df[f'load_ma{w}']  = df['load_discharge_delta'].rolling(w,1).mean()
    df[f'load_std{w}'] = df['load_discharge_delta'].rolling(w,1).std()
    df[f'load_min{w}'] = df['load_discharge_delta'].rolling(w,1).min()
    df[f'load_max{w}'] = df['load_discharge_delta'].rolling(w,1).max()
    df[f'load_q25{w}'] = df['load_discharge_delta'].rolling(w,1).quantile(0.25)
    df[f'load_q75{w}'] = df['load_discharge_delta'].rolling(w,1).quantile(0.75)

# 周期性特征增强
df['load_lag_week'] = df['load_discharge_delta'].shift(96*7)  # 同一周期
df['load_lag_day'] = df['load_discharge_delta'].shift(96)     # 同一时刻昨天
df['load_ma_week'] = df['load_discharge_delta'].rolling(96*7,1).mean()

# 差分特征
df['load_diff1'] = df['load_discharge_delta'].diff(1)
df['load_diff24'] = df['load_discharge_delta'].diff(24)
df['load_diff96'] = df['load_discharge_delta'].diff(96)

# 比率特征
df['load_ratio_ma24'] = df['load_discharge_delta'] / (df['load_ma24'] + 1e-6)
df['load_ratio_ma96'] = df['load_discharge_delta'] / (df['load_ma96'] + 1e-6)

# 时间交互特征
df['hour_load_interaction'] = df['hour'] * df['load_discharge_delta']
df['weekday_load_interaction'] = df['weekday'] * df['load_discharge_delta']
df['is_peak_load_interaction'] = df['is_peak'] * df['load_discharge_delta']

# prev_load (teacher forcing)
df['prev_load'] = df['load_discharge_delta'].shift(1)

df = df.fillna(method='ffill').fillna(method='bfill').dropna()

# 清理无穷大和异常值
print("INFO: Cleaning infinite values and outliers...")
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
    
    # 最终数据验证 - 只执行一次
    print("INFO: Validating data quality...")
    for col in set(enc_cols + dec_cols + ['load_discharge_delta', 'total_active_power']):
        if col in data.columns:
            if data[col].isnull().any():
                print(f"WARNING: {col} has {data[col].isnull().sum()} null values, filling with median")
                data[col] = data[col].fillna(data[col].median())
            if np.isinf(data[col]).any():
                print(f"WARNING: {col} has infinite values, replacing with boundary values")
                data[col] = data[col].replace([np.inf, -np.inf], [data[col].quantile(0.99), data[col].quantile(0.01)])
    
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

# ---------- 增强模型结构（微调专用） - 双输出 ----------
class EnhancedEncDec(nn.Module):
    def __init__(self, base_model, hid, drop):
        super().__init__()
        # 复制基础模型的参数
        self.enc = base_model.enc
        self.dec = base_model.dec
        self.dp = base_model.dp
        
        # 添加增强层
        self.bn = nn.BatchNorm1d(hid)
        # 使用更复杂的输出层
        self.fc_energy_enhanced = nn.Sequential(
            nn.Linear(hid, hid//2),
            nn.ReLU(),
            nn.Dropout(drop/2),
            nn.Linear(hid//2, 1)
        )
        self.fc_power_enhanced = nn.Sequential(
            nn.Linear(hid, hid//2),
            nn.ReLU(),
            nn.Dropout(drop/2),
            nn.Linear(hid//2, 1)
        )
        
        # 保留原始输出层用于残差连接
        self.fc_energy_orig = base_model.fc_energy
        self.fc_power_orig = base_model.fc_power
        
    def forward(self, xe, xd):
        _, (h,c) = self.enc(xe)
        out,_ = self.dec(xd,(h,c))
        out_dp = self.dp(out)
        
        # 应用批归一化（需要调整维度）
        batch_size, seq_len, hidden_size = out_dp.shape
        out_bn = self.bn(out_dp.transpose(1, 2)).transpose(1, 2)
        
        # 增强预测 + 原始预测的残差连接
        energy_enhanced = self.fc_energy_enhanced(out_bn).squeeze(-1)
        power_enhanced = self.fc_power_enhanced(out_bn).squeeze(-1)
        
        energy_orig = self.fc_energy_orig(out_dp).squeeze(-1)
        power_orig = self.fc_power_orig(out_dp).squeeze(-1)
        
        # 残差连接：增强预测 + 0.3 * 原始预测
        energy_pred = energy_enhanced + 0.3 * energy_orig
        power_pred = power_enhanced + 0.3 * power_orig
        
        return energy_pred, power_pred

# ---------- 智能权重策略 - 基于MAPE分析优化 ----------
class WeightedL1(nn.Module):
    def __init__(self, fut, device):
        super().__init__()
        # 智能权重策略：基于实际MAPE表现调整
        # 分析发现D3和D6表现最好，D2和D5表现最差
        # 新策略：适度优化差的天数，同时保持好天数的优势
        w = np.concatenate([
            np.ones(96)*1.8,      # Day1: 适中权重 (21.45% MAPE)
            np.ones(96)*3.0,      # Day2: 高权重但不过度 (52.03% MAPE)
            np.ones(96)*0.6,      # Day3: 低权重保持优势 (12.44% MAPE)
            np.ones(96)*2.2,      # Day4: 适中偏高权重 (32.94% MAPE)
            np.ones(96)*2.8,      # Day5: 高权重但不过度 (26.68% MAPE)
            np.ones(96)*0.6,      # Day6: 低权重保持优势 (12.45% MAPE)
            np.ones(96)*1.5       # Day7: 适中权重 (15.46% MAPE)
        ])
        self.register_buffer("w", torch.tensor(w, dtype=torch.float32, device=device))
    def forward(self, pred, target):
        # 使用Huber Loss + 自适应权重
        diff = torch.abs(pred - target)
        huber_delta = 0.08  # 稍微降低阈值，更敏感
        loss = torch.where(diff < huber_delta, 
                          0.5 * diff ** 2, 
                          huber_delta * (diff - 0.5 * huber_delta))
        return torch.mean(self.w * loss)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 先加载基础模型
base_model = EncDec(len(enc_cols), len(dec_cols), 128, .24).to(device)
base_model.load_state_dict(torch.load(f'{ROOT}/model_weighted.pth', map_location=device))

# 创建增强模型
model = EnhancedEncDec(base_model, 128, .24).to(device)

# ------ 只微调增强层 + Decoder ------
# 冻结编码器
for p in model.enc.parameters():
    p.requires_grad_(False)
# 冻结原始输出层
for p in model.fc_energy_orig.parameters():
    p.requires_grad_(False)
for p in model.fc_power_orig.parameters():
    p.requires_grad_(False)

criterion = WeightedL1(FUT_STEPS, device)
# 使用AdamW优化器，添加权重衰减 - 优化版本
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                             lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999))  # 提高学习率和正则化
# 使用余弦退火调度器，更平滑的学习率衰减
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1, eta_min=1e-6)

best, wait = 1e9, 0
print(f"\nSTART: Fine-tuning station={STATION_ID}  samples={len(tr_ds)} / {len(va_ds)}")

# 渐进式训练：前期关注短期预测，后期关注长期预测
def get_progressive_weight(epoch, total_epochs):
    """智能渐进式权重：基于MAPE表现优化"""
    progress = epoch / total_epochs
    if progress < 0.25:  # 早期：建立基础，平衡所有天数
        return np.concatenate([
            np.ones(96)*1.5, np.ones(96)*2.0, np.ones(96)*1.0,
            np.ones(96)*1.5, np.ones(96)*2.0, np.ones(96)*1.0, np.ones(96)*1.2
        ])
    elif progress < 0.6:  # 中期：重点优化问题天数，保护优势天数
        return np.concatenate([
            np.ones(96)*1.8, np.ones(96)*3.2, np.ones(96)*0.5,
            np.ones(96)*2.5, np.ones(96)*3.0, np.ones(96)*0.5, np.ones(96)*1.6
        ])
    else:  # 后期：精细调优，使用最终权重
        return np.concatenate([
            np.ones(96)*1.8, np.ones(96)*3.0, np.ones(96)*0.6,
            np.ones(96)*2.2, np.ones(96)*2.8, np.ones(96)*0.6, np.ones(96)*1.5
        ])

# 添加MAPE监控
def evaluate_day_mape(model, va_loader, device):
    """评估各天的MAPE"""
    model.eval()
    all_pred_energy, all_pred_power = [], []
    all_true_energy, all_true_power = [], []
    
    with torch.no_grad():
        for xe, xd, yy_energy, yy_power in va_loader:
            xe, xd = xe.to(device), xd.to(device)
            pred_energy, pred_power = model(xe, xd)
            all_pred_energy.append(pred_energy.cpu().numpy())
            all_pred_power.append(pred_power.cpu().numpy())
            all_true_energy.append(yy_energy.numpy())
            all_true_power.append(yy_power.numpy())
    
    pred_energy = np.concatenate(all_pred_energy, axis=0).flatten()
    pred_power = np.concatenate(all_pred_power, axis=0).flatten()
    true_energy = np.concatenate(all_true_energy, axis=0).flatten()
    true_power = np.concatenate(all_true_power, axis=0).flatten()
    
    # 计算各天MAPE
    day_mapes_energy, day_mapes_power = [], []
    for d in range(7):
        s, e = d*96, (d+1)*96
        if s < len(true_energy) and e <= len(true_energy):
            te, pe = true_energy[s:e], pred_energy[s:e]
            tp, pp = true_power[s:e], pred_power[s:e]
            
            # 安全MAPE计算
            mask_e = (np.abs(te) > 1e-3) & np.isfinite(te) & np.isfinite(pe)
            mask_p = (np.abs(tp) > 1e-3) & np.isfinite(tp) & np.isfinite(pp)
            
            if mask_e.sum() > 0:
                mape_e = np.mean(np.abs((te[mask_e] - pe[mask_e]) / te[mask_e])) * 100
                day_mapes_energy.append(min(mape_e, 200.0))
            else:
                day_mapes_energy.append(0.0)
                
            if mask_p.sum() > 0:
                mape_p = np.mean(np.abs((tp[mask_p] - pp[mask_p]) / tp[mask_p])) * 100
                day_mapes_power.append(min(mape_p, 200.0))
            else:
                day_mapes_power.append(0.0)
    
    return day_mapes_energy, day_mapes_power

for ep in range(1, EPOCHS+1):
    # 动态调整权重
    if ep % 12 == 1:  # 每12轮更新一次权重
        new_weights = get_progressive_weight(ep, EPOCHS)
        criterion.w.data = torch.tensor(new_weights, dtype=torch.float32, device=device)
        if ep == 1:
            print(f"TARGET: Epoch {ep}: Updated weights = {new_weights[:7]}")  # 只显示前7个权重的代表值
    
    model.train(); tr=0; tr_energy=0; tr_power=0
    for xe,xd,yy_energy,yy_power in tr_loader:
        xe,xd,yy_energy,yy_power = xe.to(device), xd.to(device), yy_energy.to(device), yy_power.to(device)
        
        # 数据增强：添加小幅噪声提高泛化能力
        if ep > 10:  # 前10轮不加噪声，确保基础学习
            noise_scale = 0.01 * (1 - ep/EPOCHS)  # 噪声随训练减少
            xe = xe + torch.randn_like(xe) * noise_scale
            xd = xd + torch.randn_like(xd) * noise_scale
        
        optimizer.zero_grad()
        pred_energy, pred_power = model(xe,xd)
        
        # 计算损失
        loss_energy = criterion(pred_energy, yy_energy)
        loss_power = criterion(pred_power, yy_power)
        
        # 添加一致性损失：相邻时间点的预测应该平滑
        if ep > 20:  # 后期训练加入平滑约束
            smooth_loss_energy = torch.mean(torch.abs(pred_energy[:, 1:] - pred_energy[:, :-1]))
            smooth_loss_power = torch.mean(torch.abs(pred_power[:, 1:] - pred_power[:, :-1]))
            loss = loss_energy + loss_power + 0.1 * (smooth_loss_energy + smooth_loss_power)
        else:
            loss = loss_energy + loss_power
            
        loss.backward()
        
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        tr += loss.item()
        tr_energy += loss_energy.item()
        tr_power += loss_power.item()
    tr /= len(tr_loader)
    tr_energy /= len(tr_loader)
    tr_power /= len(tr_loader)

    model.eval(); va=0; va_energy=0; va_power=0
    with torch.no_grad():
        for xe,xd,yy_energy,yy_power in va_loader:
            xe,xd,yy_energy,yy_power = xe.to(device), xd.to(device), yy_energy.to(device), yy_power.to(device)
            pred_energy, pred_power = model(xe,xd)
            loss_energy = criterion(pred_energy, yy_energy)
            loss_power = criterion(pred_power, yy_power)
            va += (loss_energy + loss_power).item()
            va_energy += loss_energy.item()
            va_power += loss_power.item()
    va /= len(va_loader)
    va_energy /= len(va_loader)
    va_power /= len(va_loader)
    
    scheduler.step()  # CosineAnnealingWarmRestarts不需要传入loss

    log = f'E{ep:03d}  tr {tr:.4f}(E:{tr_energy:.3f},P:{tr_power:.3f})  va {va:.4f}(E:{va_energy:.3f},P:{va_power:.3f})  lr {optimizer.param_groups[0]["lr"]:.2e}'
    
    # 每20轮评估一次各天MAPE
    if ep % 20 == 0:
        day_mapes_e, day_mapes_p = evaluate_day_mape(model, va_loader, device)
        if len(day_mapes_e) >= 7:
            log += f'\n    MONITOR: Day MAPE: E[{day_mapes_e[1]:.1f},{day_mapes_e[2]:.1f},{day_mapes_e[4]:.1f}] P[{day_mapes_p[1]:.1f},{day_mapes_p[2]:.1f},{day_mapes_p[4]:.1f}]'
    
    if va < best:
        best = va; wait = 0
        folder = f'{ROOT}/model_{STATION_ID}'; os.makedirs(folder, exist_ok=True)
        torch.save(model.state_dict(), f'{folder}/model_optimized_{STATION_ID}.pth')
        log += '  SAVE'
    else:
        wait += 1
        if wait >= 25:  # 增加早停耐心，给智能权重更多时间
            log += '  (early stop)'; print(log); break
    print(log)

# ---------- 评估 - 双输出 ----------
def day_mape(t,p):
    res=[]
    for d in range(7):
        s,e=d*96,(d+1)*96
        t0,t1=t[s:e],p[s:e]
        # 更严格的处理：过滤掉异常值
        mask = (np.abs(t0) > 1e-3) & np.isfinite(t0) & np.isfinite(t1)
        if mask.sum() == 0:
            res.append(0.0)  # 如果没有有效数据，返回0
        else:
            t0_filtered = t0[mask]
            t1_filtered = t1[mask]
            # 使用绝对值确保分母为正
            t0_filtered = np.where(np.abs(t0_filtered) < 1e-3, 
                                 np.sign(t0_filtered) * 1e-3, t0_filtered)
            mape = np.mean(np.abs((t0_filtered - t1_filtered) / t0_filtered)) * 100
            # 限制MAPE的最大值，避免极端情况
            res.append(min(mape, 1000.0))
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

    # 安全的MAPE计算
    def safe_mape(y_true, y_pred):
        mask = (np.abs(y_true) > 1e-3) & np.isfinite(y_true) & np.isfinite(y_pred)
        if mask.sum() == 0:
            return 0.0
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        y_true_filtered = np.where(np.abs(y_true_filtered) < 1e-3, 
                                 np.sign(y_true_filtered) * 1e-3, y_true_filtered)
        mape = np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100
        return min(mape, 1000.0)
    
    mape_energy = safe_mape(true_energy, pred_energy)
    mape_power = safe_mape(true_power, pred_power)
    
    rmse_energy = np.sqrt(mean_squared_error(true_energy,pred_energy))
    rmse_power = np.sqrt(mean_squared_error(true_power,pred_power))

    dm_energy = day_mape(true_energy,pred_energy)
    dm_power = day_mape(true_power,pred_power)
    
    dm_energy_str = ' | '.join([f'D{i+1}:{m:.2f}%' for i,m in enumerate(dm_energy)])
    dm_power_str = ' | '.join([f'D{i+1}:{m:.2f}%' for i,m in enumerate(dm_power)])

print(f'\nRESULT: [{STATION_ID}] Energy 7-day MAPE={mape_energy:.2f}%  RMSE={rmse_energy:.2f}')
print(f'RESULT: [{STATION_ID}] Power  7-day MAPE={mape_power:.2f}%   RMSE={rmse_power:.2f}')
print(f'RESULT: [{STATION_ID}] Energy Day-wise MAPE  {dm_energy_str}')
print(f'RESULT: [{STATION_ID}] Power  Day-wise MAPE  {dm_power_str}')
print(f'\nSUCCESS: Fine-tuning completed. Model saved to output_pytorch/model_{STATION_ID}/model_optimized_{STATION_ID}.pth')
