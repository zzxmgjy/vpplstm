# =========================================================
#  Single Station Encoder-Decoder Model for Power Prediction
#  Predicts: total_active_power and not_use_power for 7 days
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
import argparse

warnings.filterwarnings("ignore")

# ---------- ⚙ Configuration ----------
CFG = dict(
    past_steps   = 96*7,          # 7 days history (15min intervals)
    future_steps = 96*7,          # 7 days prediction (672 points)
    hidden_dim   = 512,           
    num_layers   = 1,             
    drop_rate    = .4,            
    batch_size   = 256,            
    epochs       = 200,           
    patience     = 30,            
    lr           = 0.0001,          
    top_k        = 90,            # Reduced features since no energy fields
    lgb_rounds   = 500,
    use_stl      = True,          
    power_weight = 1.0,           # Main prediction weight
    not_use_power_weight = 0.7    # Secondary prediction weight
)

def main(station_id=None, data_file='merged_station_test.csv'):
    t0 = time.time()
    
    # =========================================================
    # 1. Read and filter data for specific station
    # =========================================================
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file, parse_dates=['ts'], dtype={'station_ref_id': str})
    df = df.sort_values(['station_ref_id', 'ts'])
    
    # If station_id specified, filter for that station only
    if station_id:
        df = df[df['station_ref_id'] == station_id].copy()
        if len(df) == 0:
            raise ValueError(f"No data found for station {station_id}")
        print(f"Training model for station: {station_id}")
        output_dir = f"models/station_{station_id}"
    else:
        # Use first station if none specified
        station_id = df['station_ref_id'].iloc[0]
        df = df[df['station_ref_id'] == station_id].copy()
        print(f"Training model for station: {station_id} (first station)")
        output_dir = f"models/station_{station_id}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check required fields
    required_fields = ['ts', 'total_active_power', 'station_ref_id']
    missing_fields = [f for f in required_fields if f not in df.columns]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    # Create not_use_power if not exists (assuming it's non-controllable load)
    if 'not_use_power' not in df.columns:
        # Estimate not_use_power as a portion of total_active_power
        df['not_use_power'] = df['total_active_power'] * 0.7  # Assume 70% is non-controllable
        print("WARNING: not_use_power field created as 70% of total_active_power")
    
    # Clean power data
    df['total_active_power'] = df['total_active_power'].clip(lower=0)
    df['not_use_power'] = df['not_use_power'].clip(lower=0)
    
    # Rename ts to energy_date for compatibility
    df = df.rename(columns={'ts': 'energy_date'})
    
    # Handle optional fields with defaults
    optional_fields = {
        'temp': 25.0,           
        'humidity': 60.0,       
        'windSpeed': 5.0,       
        'code': 999             
    }
    
    for field, default_value in optional_fields.items():
        if field not in df.columns:
            df[field] = default_value
            print(f"WARNING: Field '{field}' missing, set default: {default_value}")
    
    df['code'] = df['code'].fillna(999).astype(int)
    
    # One-hot encoding for code
    df = pd.concat([df, pd.get_dummies(df['code'].astype(str), prefix='code')], axis=1)
    
    # =========================================================
    # 2. Time/Weather/Holiday Features
    # =========================================================
    cn_holidays = holidays.country_holidays('CN')
    
    def make_is_peak(ts):
        h, mi = ts.dt.hour, ts.dt.minute
        return ((h>8)|((h==8)&(mi>=30))) & ((h<17)|((h==17)&(mi<=30)))
    
    def enrich_features(d: pd.DataFrame):
        d['hour']    = d['energy_date'].dt.hour
        d['minute']  = d['energy_date'].dt.minute
        d['weekday'] = d['energy_date'].dt.weekday
        d['month']   = d['energy_date'].dt.month
        d['day']     = d['energy_date'].dt.day
        
        # Weather features
        if 'temp' in d.columns and 'humidity' in d.columns:
            d['dew_point']  = d['temp'] - (100-d['humidity'])/5
            d['feels_like'] = d['temp'] + 0.33*d['humidity'] - 4
            for k in [1,24]:
                d[f'temp_diff{k}'] = d['temp'].diff(k)
        else:
            d['dew_point'] = 20.0
            d['feels_like'] = 25.0
            d['temp_diff1'] = 0.0
            d['temp_diff24'] = 0.0
        
        # Cyclical features
        d['sin_hour'] = np.sin(2*np.pi*(d['hour']+d['minute']/60)/24)
        d['cos_hour'] = np.cos(2*np.pi*(d['hour']+d['minute']/60)/24)
        d['sin_wday'] = np.sin(2*np.pi*d['weekday']/7)
        d['cos_wday'] = np.cos(2*np.pi*d['weekday']/7)
        
        # Calendar features
        d['is_holiday'] = d['energy_date'].isin(cn_holidays).astype(int)
        d['is_work']    = ((d['weekday']<5)&(~d['energy_date'].isin(cn_holidays))).astype(int)
        d['is_peak']    = make_is_peak(d['energy_date']).astype(int)
        
        for lag in [1,2,3]:
            d[f'before_holiday_{lag}'] = d['energy_date'].shift(-lag).isin(cn_holidays).astype(int)
            d[f'after_holiday_{lag}']  = d['energy_date'].shift(lag ).isin(cn_holidays).astype(int)
        
        d['is_month_begin'] = (d['day']<=3).astype(int)
        d['is_month_end']   = d['energy_date'].dt.is_month_end.astype(int)
        return d
    
    df = enrich_features(df)
    
    # =========================================================
    # 3. Power-based Features (NO ENERGY FIELDS)
    # =========================================================
    # Lag features for power
    for lag in [1,2,4,8,12,24,48,96]:
        df[f'power_lag{lag}'] = df['total_active_power'].shift(lag)
        df[f'not_use_power_lag{lag}'] = df['not_use_power'].shift(lag)
    
    # Rolling statistics for power
    for w in [4,8,12,24,48,96]:
        df[f'power_ma{w}']  = df['total_active_power'].rolling(w,1).mean()
        df[f'power_std{w}'] = df['total_active_power'].rolling(w,1).std()
        df[f'not_use_power_ma{w}']  = df['not_use_power'].rolling(w,1).mean()
        df[f'not_use_power_std{w}'] = df['not_use_power'].rolling(w,1).std()
    
    # Previous power for teacher forcing
    df['prev_power'] = df['total_active_power'].shift(1)
    df['prev_not_use_power'] = df['not_use_power'].shift(1)
    
    # Weekly/bi-weekly lags
    for lag in [7*96, 14*96]:  
        if len(df) > lag:
            df[f'power_lag{lag//96}d'] = df['total_active_power'].shift(lag)
            df[f'not_use_power_lag{lag//96}d'] = df['not_use_power'].shift(lag)
    
    # Power ratio features
    df['power_ratio'] = df['not_use_power'] / (df['total_active_power'] + 1e-6)
    df['power_ratio'] = df['power_ratio'].clip(0, 1)  # Ratio should be 0-1
    
    # =========================================================
    # 4. STL Decomposition + Quantile Normalization
    # =========================================================
    def add_advanced_features(g):
        if CFG['use_stl'] and len(g)>=96*14:
            # STL for total power
            res_power = STL(g['total_active_power'], period=96*7, robust=True).fit()
            g['power_trend']    = res_power.trend
            g['power_seasonal'] = res_power.seasonal
            g['power_resid']    = res_power.resid
            
            # STL for not_use_power
            res_not_use = STL(g['not_use_power'], period=96*7, robust=True).fit()
            g['not_use_power_trend']    = res_not_use.trend
            g['not_use_power_seasonal'] = res_not_use.seasonal
            g['not_use_power_resid']    = res_not_use.resid
        else:
            g[['power_trend','power_seasonal','power_resid']] = np.nan
            g[['not_use_power_trend','not_use_power_seasonal','not_use_power_resid']] = np.nan
        
        # Quantile normalization
        g['power_q10_48'] = g['total_active_power'].rolling(48,1).quantile(.1)
        g['power_q90_48'] = g['total_active_power'].rolling(48,1).quantile(.9)
        g['power_norm_48']= g['total_active_power'] / (g['power_q90_48'] + 1e-6)
        
        g['not_use_power_q10_48'] = g['not_use_power'].rolling(48,1).quantile(.1)
        g['not_use_power_q90_48'] = g['not_use_power'].rolling(48,1).quantile(.9)
        g['not_use_power_norm_48']= g['not_use_power'] / (g['not_use_power_q90_48'] + 1e-6)
        
        return g
    
    df = add_advanced_features(df)
    
    # =========================================================
    # 5. Data Cleaning
    # =========================================================
    df = df.fillna(method='ffill').fillna(method='bfill')
    df = df.dropna(subset=['total_active_power', 'not_use_power'])
    
    # Fill remaining NaN values
    for col in ['power_trend','power_seasonal','power_resid',
                'not_use_power_trend','not_use_power_seasonal','not_use_power_resid',
                'power_q10_48','power_q90_48','power_norm_48',
                'not_use_power_q10_48','not_use_power_q90_48','not_use_power_norm_48',
                'prev_power','prev_not_use_power']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Clean infinite and extreme values
    print("Cleaning infinite and extreme values...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)
            if df[col].std() > 0:
                upper_bound = df[col].quantile(0.999)
                lower_bound = df[col].quantile(0.001)
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    print(f"Data rows: {len(df):,}")
    
    # =========================================================
    # 6. Feature Selection
    # =========================================================
    # Exclude energy fields and target fields
    ENC_FULL = [c for c in df.columns
                if c not in ['energy_date','station_ref_id','total_active_power','not_use_power',
                           'forward_total_active_energy','backward_total_active_energy',
                           'load_discharge_delta','label'] and 'energy' not in c.lower()]
    
    DEC_FULL = [c for c in ENC_FULL if (not c.startswith('power_') and not c.startswith('not_use_power_'))] + \
               ['prev_power', 'prev_not_use_power']
    
    # =========================================================
    # 7. LightGBM Feature Selection
    # =========================================================
    if len(df) >= 1000:
        sample_df = df.sample(frac=0.3, random_state=42)
    else:
        sample_df = df.copy()
        
    X, y = sample_df[ENC_FULL], sample_df['total_active_power']
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
    
    # Ensure previous power features are included
    for prev_col in ['prev_power', 'prev_not_use_power']:
        if prev_col not in ENC_COLS and prev_col in df.columns:
            ENC_COLS.append(prev_col)
    
    DEC_COLS=[c for c in ENC_COLS if not c.startswith('power_') and not c.startswith('not_use_power_')]
    for prev_col in ['prev_power', 'prev_not_use_power']:
        if prev_col not in DEC_COLS and prev_col in df.columns:
            DEC_COLS.append(prev_col)
    
    print(f"Selected {len(ENC_COLS)} encoder, {len(DEC_COLS)} decoder features")
    
    # =========================================================
    # 8. Create Dataset - Dual Output
    # =========================================================
    def make_dataset(data, past, fut):
        Xp, Xf, Y_power, Y_not_use_power = [], [], [], []
        sc_e, sc_d, sc_y_power, sc_y_not_use_power = StandardScaler(), StandardScaler(), StandardScaler(), StandardScaler()
        
        # Data validation
        print("Validating data quality...")
        for col in ENC_COLS + DEC_COLS + ['total_active_power', 'not_use_power']:
            if col in data.columns:
                if data[col].isnull().any():
                    print(f"WARNING: {col} has {data[col].isnull().sum()} null values, filling with median")
                    data[col] = data[col].fillna(data[col].median())
                if np.isinf(data[col]).any():
                    print(f"WARNING: {col} has infinite values, replacing with boundary values")
                    data[col] = data[col].replace([np.inf, -np.inf], 
                                                [data[col].quantile(0.99), data[col].quantile(0.01)])
        
        e_all = sc_e.fit_transform(data[ENC_COLS])
        d_all = sc_d.fit_transform(data[DEC_COLS])
        y_power_all = sc_y_power.fit_transform(data[['total_active_power']])
        y_not_use_power_all = sc_y_not_use_power.fit_transform(data[['not_use_power']])
        
        e_df = pd.DataFrame(e_all, columns=ENC_COLS, index=data.index)
        e_df['y_power'] = y_power_all
        e_df['y_not_use_power'] = y_not_use_power_all
        d_df = pd.DataFrame(d_all, columns=DEC_COLS, index=data.index)
        
        # Since we're working with single station, no groupby needed
        if len(data) < past + fut:
            raise ValueError(f"Insufficient data: {len(data)} < {past + fut}")
            
        e_arr = e_df[ENC_COLS].values
        d_arr = d_df[DEC_COLS].values
        y_power_arr = e_df['y_power'].values
        y_not_use_power_arr = e_df['y_not_use_power'].values
        
        for i in range(len(data) - past - fut + 1):
            Xp.append(e_arr[i:i+past])
            Xf.append(d_arr[i+past:i+past+fut])
            Y_power.append(y_power_arr[i+past:i+past+fut])
            Y_not_use_power.append(y_not_use_power_arr[i+past:i+past+fut])
        
        return (np.array(Xp, np.float32), np.array(Xf, np.float32),
                np.array(Y_power, np.float32), np.array(Y_not_use_power, np.float32),
                sc_e, sc_d, sc_y_power, sc_y_not_use_power)
    
    Xp, Xf, Y_power, Y_not_use_power, sc_e, sc_d, sc_y_power, sc_y_not_use_power = \
        make_dataset(df, CFG['past_steps'], CFG['future_steps'])
    
    print(f"Samples: {len(Xp)}")
    
    if len(Xp) == 0:
        raise ValueError("No samples generated. Check data length and parameters.")
    
    spl = int(.8 * len(Xp))
    tr_ds = TensorDataset(torch.from_numpy(Xp[:spl]), torch.from_numpy(Xf[:spl]),
                         torch.from_numpy(Y_power[:spl]), torch.from_numpy(Y_not_use_power[:spl]))
    va_ds = TensorDataset(torch.from_numpy(Xp[spl:]), torch.from_numpy(Xf[spl:]),
                         torch.from_numpy(Y_power[spl:]), torch.from_numpy(Y_not_use_power[spl:]))
    
    tr_loader = DataLoader(tr_ds, batch_size=CFG['batch_size'], shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=CFG['batch_size'], shuffle=False)
    
    # =========================================================
    # 9. Model Definition
    # =========================================================
    class EncDecPowerModel(nn.Module):
        def __init__(self, d_enc, d_dec, hid, drop, num_layers=2):
            super().__init__()
            self.enc = nn.LSTM(d_enc, hid, num_layers=num_layers, batch_first=True, 
                              dropout=drop if num_layers>1 else 0)
            self.dec = nn.LSTM(d_dec, hid, num_layers=num_layers, batch_first=True, 
                              dropout=drop if num_layers>1 else 0)
            self.dp = nn.Dropout(drop)
            
            # Enhanced middle layers
            self.fc_mid = nn.Sequential(
                nn.Linear(hid, hid//2),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(hid//2, hid//4),
                nn.ReLU(),
                nn.Dropout(drop)
            )
            
            self.fc_power = nn.Linear(hid//4, 1)           # total_active_power prediction
            self.fc_not_use_power = nn.Linear(hid//4, 1)   # not_use_power prediction
            
        def forward(self, xe, xd):
            _, (h, c) = self.enc(xe)
            out, _ = self.dec(xd, (h, c))
            out_dp = self.dp(out)
            out_mid = self.fc_mid(out_dp)
            power_pred = self.fc_power(out_mid).squeeze(-1)
            not_use_power_pred = self.fc_not_use_power(out_mid).squeeze(-1)
            return power_pred, not_use_power_pred
    
    # Weighted L1 Loss
    class WeightedL1Loss(nn.Module):
        def __init__(self, fut, device):
            super().__init__()
            # Higher weights for days 3-4
            w = np.concatenate([
                np.ones(96*2),           # Day1-2
                np.ones(96)*1.3,         # Day3
                np.ones(96)*1.5,         # Day4
                np.ones(96*3)*1.2        # Day5-7
            ])
            self.register_buffer('w', torch.tensor(w, dtype=torch.float32, device=device))
            
        def forward(self, pred, target):
            return torch.mean(self.w * torch.abs(pred - target))
    
    # =========================================================
    # 10. Training
    # =========================================================
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {dev}")
    
    model = EncDecPowerModel(len(ENC_COLS), len(DEC_COLS), CFG['hidden_dim'], 
                            CFG['drop_rate'], CFG['num_layers']).to(dev)
    crit = WeightedL1Loss(CFG['future_steps'], dev)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=1e-5)
    sch = ReduceLROnPlateau(opt, 'min', patience=8, factor=.7)
    
    best = 1e9
    wait = 0
    print("Training...")
    
    for ep in range(1, CFG['epochs'] + 1):
        model.train()
        tr = 0
        for xe, xd, yy_power, yy_not_use_power in tr_loader:
            xe, xd, yy_power, yy_not_use_power = xe.to(dev), xd.to(dev), yy_power.to(dev), yy_not_use_power.to(dev)
            opt.zero_grad()
            pred_power, pred_not_use_power = model(xe, xd)
            loss_power = crit(pred_power, yy_power)
            loss_not_use_power = crit(pred_not_use_power, yy_not_use_power)
            loss = CFG['power_weight'] * loss_power + CFG['not_use_power_weight'] * loss_not_use_power
            loss.backward()
            opt.step()
            tr += loss.item()
        tr /= len(tr_loader)
        
        model.eval()
        va = 0
        with torch.no_grad():
            for xe, xd, yy_power, yy_not_use_power in va_loader:
                xe, xd, yy_power, yy_not_use_power = xe.to(dev), xd.to(dev), yy_power.to(dev), yy_not_use_power.to(dev)
                pred_power, pred_not_use_power = model(xe, xd)
                loss_power = crit(pred_power, yy_power)
                loss_not_use_power = crit(pred_not_use_power, yy_not_use_power)
                va += (CFG['power_weight'] * loss_power + CFG['not_use_power_weight'] * loss_not_use_power).item()
        va /= len(va_loader)
        sch.step(va)
        
        if ep % 10 == 0:
            print(f'E{ep:03d} tr{tr:.4f} va{va:.4f}')
        
        if va < best:
            best = va
            wait = 0
            torch.save(model.state_dict(), f'{output_dir}/model_power.pth')
        else:
            wait += 1
            if wait >= CFG['patience']:
                print("Early stopping")
                break
    
    # =========================================================
    # 11. Evaluation of MAPE for last 7 days
    # =========================================================
    print("Evaluating MAPE for last 7 days prediction...")
    
    model.eval()
    all_power_preds = []
    all_not_use_power_preds = []
    all_power_targets = []
    all_not_use_power_targets = []
    
    with torch.no_grad():
        for xe, xd, yy_power, yy_not_use_power in va_loader:
            xe, xd = xe.to(dev), xd.to(dev)
            pred_power, pred_not_use_power = model(xe, xd)
            
            # Convert predictions back to original scale
            pred_power_np = sc_y_power.inverse_transform(pred_power.cpu().numpy())
            pred_not_use_power_np = sc_y_not_use_power.inverse_transform(pred_not_use_power.cpu().numpy())
            
            # Convert targets back to original scale
            target_power_np = sc_y_power.inverse_transform(yy_power.numpy())
            target_not_use_power_np = sc_y_not_use_power.inverse_transform(yy_not_use_power.numpy())
            
            all_power_preds.append(pred_power_np)
            all_not_use_power_preds.append(pred_not_use_power_np)
            all_power_targets.append(target_power_np)
            all_not_use_power_targets.append(target_not_use_power_np)
    
    # Concatenate all predictions and targets
    all_power_preds = np.concatenate(all_power_preds, axis=0)
    all_not_use_power_preds = np.concatenate(all_not_use_power_preds, axis=0)
    all_power_targets = np.concatenate(all_power_targets, axis=0)
    all_not_use_power_targets = np.concatenate(all_not_use_power_targets, axis=0)
    
    # Calculate MAPE for each day
    days = 7
    points_per_day = 96  # 15-minute intervals for 24 hours
    
    power_mape_by_day = []
    not_use_power_mape_by_day = []
    
    for day in range(days):
        start_idx = day * points_per_day
        end_idx = (day + 1) * points_per_day
        
        # Calculate MAPE for total_active_power
        day_power_preds = all_power_preds[:, start_idx:end_idx]
        day_power_targets = all_power_targets[:, start_idx:end_idx]
        # Filter out zeros in targets to avoid division by zero
        mask = day_power_targets > 1.0
        if np.any(mask):
            day_power_mape = mean_absolute_percentage_error(
                day_power_targets[mask], 
                day_power_preds[mask]
            )
            power_mape_by_day.append(day_power_mape)
        else:
            power_mape_by_day.append(np.nan)
        
        # Calculate MAPE for not_use_power
        day_not_use_power_preds = all_not_use_power_preds[:, start_idx:end_idx]
        day_not_use_power_targets = all_not_use_power_targets[:, start_idx:end_idx]
        # Filter out zeros in targets to avoid division by zero
        mask = day_not_use_power_targets > 1.0
        if np.any(mask):
            day_not_use_power_mape = mean_absolute_percentage_error(
                day_not_use_power_targets[mask], 
                day_not_use_power_preds[mask]
            )
            not_use_power_mape_by_day.append(day_not_use_power_mape)
        else:
            not_use_power_mape_by_day.append(np.nan)
    
    # Calculate overall MAPE
    mask = all_power_targets > 1.0
    overall_power_mape = mean_absolute_percentage_error(
        all_power_targets[mask], 
        all_power_preds[mask]
    ) if np.any(mask) else np.nan
    
    mask = all_not_use_power_targets > 1.0
    overall_not_use_power_mape = mean_absolute_percentage_error(
        all_not_use_power_targets[mask], 
        all_not_use_power_preds[mask]
    ) if np.any(mask) else np.nan
    
    # Print MAPE results
    print("\nMAPE Evaluation Results:")
    print("------------------------")
    print("total_active_power MAPE by day:")
    for day, mape in enumerate(power_mape_by_day, 1):
        print(f"  Day {day}: {mape:.2%}")
    print(f"Overall total_active_power MAPE: {overall_power_mape:.2%}")
    
    print("\nnot_use_power MAPE by day:")
    for day, mape in enumerate(not_use_power_mape_by_day, 1):
        print(f"  Day {day}: {mape:.2%}")
    print(f"Overall not_use_power MAPE: {overall_not_use_power_mape:.2%}")
    
    # Save MAPE results
    mape_results = {
        'power_mape_by_day': power_mape_by_day,
        'not_use_power_mape_by_day': not_use_power_mape_by_day,
        'overall_power_mape': overall_power_mape,
        'overall_not_use_power_mape': overall_not_use_power_mape
    }
    joblib.dump(mape_results, f'{output_dir}/mape_results.pkl')
    
    # =========================================================
    # 12. Save Model Components
    # =========================================================
    joblib.dump(sc_e, f'{output_dir}/scaler_enc.pkl')
    joblib.dump(sc_d, f'{output_dir}/scaler_dec.pkl')
    joblib.dump(sc_y_power, f'{output_dir}/scaler_y_power.pkl')
    joblib.dump(sc_y_not_use_power, f'{output_dir}/scaler_y_not_use_power.pkl')
    joblib.dump(ENC_COLS, f'{output_dir}/enc_cols.pkl')
    joblib.dump(DEC_COLS, f'{output_dir}/dec_cols.pkl')
    joblib.dump(CFG, f'{output_dir}/config.pkl')
    
    # Save station info
    station_info = {
        'station_id': station_id,
        'data_points': len(df),
        'training_samples': len(Xp),
        'features_count': len(ENC_COLS)
    }
    joblib.dump(station_info, f'{output_dir}/station_info.pkl')
    
    print(f"Model saved to {output_dir}/")
    print(f"Station: {station_id}")
    print(f"Data points: {len(df):,}")
    print(f"Training samples: {len(Xp):,}")
    print(f"Total time: {time.time()-t0:.1f}s")
    
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train single station power prediction model')
    parser.add_argument('--station_id', type=str, help='Station ID to train model for')
    parser.add_argument('--data_file', type=str, default='merged_station_test.csv', 
                       help='Input data file')
    
    args = parser.parse_args()
    
    try:
        output_dir = main(args.station_id, args.data_file)
        print(f"Training completed successfully! Model saved to: {output_dir}")
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise