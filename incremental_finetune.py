# =========================================================
#  Incremental Fine-tuning for Single Station Model
#  Fine-tunes existing model with new incremental data
# =========================================================
import os, warnings, gc, time
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import joblib
import argparse
import holidays
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings("ignore")

# Fine-tuning configuration
FINETUNE_CFG = dict(
    batch_size   = 32,            # Smaller batch for fine-tuning
    epochs       = 50,            # Fewer epochs for fine-tuning
    patience     = 15,            # Early stopping patience
    lr           = 1e-5,          # Lower learning rate for fine-tuning
    power_weight = 1.0,           
    not_use_power_weight = 0.8    
)

class EncDecPowerModel(nn.Module):
    """Same model architecture as in training script"""
    def __init__(self, d_enc, d_dec, hid, drop, num_layers=2):
        super().__init__()
        self.enc = nn.LSTM(d_enc, hid, num_layers=num_layers, batch_first=True, 
                          dropout=drop if num_layers>1 else 0)
        self.dec = nn.LSTM(d_dec, hid, num_layers=num_layers, batch_first=True, 
                          dropout=drop if num_layers>1 else 0)
        self.dp = nn.Dropout(drop)
        
        self.fc_mid = nn.Sequential(
            nn.Linear(hid, hid//2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hid//2, hid//4),
            nn.ReLU(),
            nn.Dropout(drop)
        )
        
        self.fc_power = nn.Linear(hid//4, 1)           
        self.fc_not_use_power = nn.Linear(hid//4, 1)   
        
    def forward(self, xe, xd):
        _, (h, c) = self.enc(xe)
        out, _ = self.dec(xd, (h, c))
        out_dp = self.dp(out)
        out_mid = self.fc_mid(out_dp)
        power_pred = self.fc_power(out_mid).squeeze(-1)
        not_use_power_pred = self.fc_not_use_power(out_mid).squeeze(-1)
        return power_pred, not_use_power_pred

class WeightedL1Loss(nn.Module):
    """Same loss function as in training script"""
    def __init__(self, fut, device):
        super().__init__()
        w = np.concatenate([
            np.ones(96*2),           # Day1-2
            np.ones(96)*1.3,         # Day3
            np.ones(96)*1.5,         # Day4
            np.ones(96*3)*1.2        # Day5-7
        ])
        self.register_buffer('w', torch.tensor(w, dtype=torch.float32, device=device))
        
    def forward(self, pred, target):
        return torch.mean(self.w * torch.abs(pred - target))

def make_is_peak(ts):
    h, mi = ts.dt.hour, ts.dt.minute
    return ((h>8)|((h==8)&(mi>=30))) & ((h<17)|((h==17)&(mi<=30)))

def enrich_features(d: pd.DataFrame):
    """Same feature engineering as in training script"""
    cn_holidays = holidays.country_holidays('CN')
    
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

def add_power_features(df):
    """Add power-based features"""
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
    df['power_ratio'] = df['power_ratio'].clip(0, 1)
    
    return df

def add_advanced_features(df, use_stl=True):
    """Add STL and quantile features"""
    if use_stl and len(df)>=96*14:
        # STL for total power
        res_power = STL(df['total_active_power'], period=96*7, robust=True).fit()
        df['power_trend']    = res_power.trend
        df['power_seasonal'] = res_power.seasonal
        df['power_resid']    = res_power.resid
        
        # STL for not_use_power
        res_not_use = STL(df['not_use_power'], period=96*7, robust=True).fit()
        df['not_use_power_trend']    = res_not_use.trend
        df['not_use_power_seasonal'] = res_not_use.seasonal
        df['not_use_power_resid']    = res_not_use.resid
    else:
        df[['power_trend','power_seasonal','power_resid']] = np.nan
        df[['not_use_power_trend','not_use_power_seasonal','not_use_power_resid']] = np.nan
    
    # Quantile normalization
    df['power_q10_48'] = df['total_active_power'].rolling(48,1).quantile(.1)
    df['power_q90_48'] = df['total_active_power'].rolling(48,1).quantile(.9)
    df['power_norm_48']= df['total_active_power'] / (df['power_q90_48'] + 1e-6)
    
    df['not_use_power_q10_48'] = df['not_use_power'].rolling(48,1).quantile(.1)
    df['not_use_power_q90_48'] = df['not_use_power'].rolling(48,1).quantile(.9)
    df['not_use_power_norm_48']= df['not_use_power'] / (df['not_use_power_q90_48'] + 1e-6)
    
    return df

def prepare_incremental_data(data_file, station_id):
    """Prepare incremental data with same preprocessing as training"""
    print(f"📊 Loading incremental data from {data_file}...")
    df = pd.read_csv(data_file, parse_dates=['ts'])
    df = df.sort_values(['station_ref_id', 'ts'])
    
    # Filter for specific station
    df = df[df['station_ref_id'] == station_id].copy()
    if len(df) == 0:
        raise ValueError(f"No data found for station {station_id}")
    
    # Create not_use_power if not exists
    if 'not_use_power' not in df.columns:
        df['not_use_power'] = df['total_active_power'] * 0.7
        print("⚠️  not_use_power field created as 70% of total_active_power")
    
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
    
    df['code'] = df['code'].fillna(999).astype(int)
    
    # One-hot encoding for code
    df = pd.concat([df, pd.get_dummies(df['code'].astype(str), prefix='code')], axis=1)
    
    # Feature engineering
    df = enrich_features(df)
    df = add_power_features(df)
    df = add_advanced_features(df)
    
    # Data cleaning
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
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)
            if df[col].std() > 0:
                upper_bound = df[col].quantile(0.999)
                lower_bound = df[col].quantile(0.001)
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df

def create_finetune_dataset(data, enc_cols, dec_cols, sc_e, sc_d, sc_y_power, sc_y_not_use_power, past_steps, future_steps):
    """Create dataset for fine-tuning using existing scalers"""
    Xp, Xf, Y_power, Y_not_use_power = [], [], [], []
    
    # Use existing scalers to transform data
    e_all = sc_e.transform(data[enc_cols])
    d_all = sc_d.transform(data[dec_cols])
    y_power_all = sc_y_power.transform(data[['total_active_power']])
    y_not_use_power_all = sc_y_not_use_power.transform(data[['not_use_power']])
    
    e_df = pd.DataFrame(e_all, columns=enc_cols, index=data.index)
    e_df['y_power'] = y_power_all
    e_df['y_not_use_power'] = y_not_use_power_all
    d_df = pd.DataFrame(d_all, columns=dec_cols, index=data.index)
    
    if len(data) < past_steps + future_steps:
        raise ValueError(f"Insufficient data for fine-tuning: {len(data)} < {past_steps + future_steps}")
        
    e_arr = e_df[enc_cols].values
    d_arr = d_df[dec_cols].values
    y_power_arr = e_df['y_power'].values
    y_not_use_power_arr = e_df['y_not_use_power'].values
    
    for i in range(len(data) - past_steps - future_steps + 1):
        Xp.append(e_arr[i:i+past_steps])
        Xf.append(d_arr[i+past_steps:i+past_steps+future_steps])
        Y_power.append(y_power_arr[i+past_steps:i+past_steps+future_steps])
        Y_not_use_power.append(y_not_use_power_arr[i+past_steps:i+past_steps+future_steps])
    
    return (np.array(Xp, np.float32), np.array(Xf, np.float32),
            np.array(Y_power, np.float32), np.array(Y_not_use_power, np.float32))

def finetune_model(model_dir, incremental_data_file, station_id):
    """Fine-tune existing model with incremental data"""
    t0 = time.time()
    
    # Load existing model components
    print(f"📂 Loading model from {model_dir}...")
    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory not found: {model_dir}")
    
    # Load configuration and components
    cfg = joblib.load(f'{model_dir}/config.pkl')
    enc_cols = joblib.load(f'{model_dir}/enc_cols.pkl')
    dec_cols = joblib.load(f'{model_dir}/dec_cols.pkl')
    sc_e = joblib.load(f'{model_dir}/scaler_enc.pkl')
    sc_d = joblib.load(f'{model_dir}/scaler_dec.pkl')
    sc_y_power = joblib.load(f'{model_dir}/scaler_y_power.pkl')
    sc_y_not_use_power = joblib.load(f'{model_dir}/scaler_y_not_use_power.pkl')
    station_info = joblib.load(f'{model_dir}/station_info.pkl')
    
    print(f"🎯 Fine-tuning model for station: {station_info['station_id']}")
    
    # Prepare incremental data
    df_new = prepare_incremental_data(incremental_data_file, station_id)
    print(f"📊 Incremental data points: {len(df_new):,}")
    
    # Ensure all required columns exist in new data
    missing_cols = []
    for col in enc_cols + dec_cols:
        if col not in df_new.columns:
            missing_cols.append(col)
    
    if missing_cols:
        print(f"⚠️  Missing columns in incremental data: {missing_cols}")
        # Fill missing columns with zeros or appropriate defaults
        for col in missing_cols:
            df_new[col] = 0
    
    # Create fine-tuning dataset
    Xp, Xf, Y_power, Y_not_use_power = create_finetune_dataset(
        df_new, enc_cols, dec_cols, sc_e, sc_d, sc_y_power, sc_y_not_use_power,
        cfg['past_steps'], cfg['future_steps']
    )
    
    print(f"🍱 Fine-tuning samples: {len(Xp)}")
    
    if len(Xp) == 0:
        raise ValueError("No samples generated for fine-tuning. Check data length and parameters.")
    
    # Create data loaders
    ft_ds = TensorDataset(torch.from_numpy(Xp), torch.from_numpy(Xf),
                         torch.from_numpy(Y_power), torch.from_numpy(Y_not_use_power))
    ft_loader = DataLoader(ft_ds, batch_size=FINETUNE_CFG['batch_size'], shuffle=True)
    
    # Load and initialize model
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {dev}")
    
    model = EncDecPowerModel(len(enc_cols), len(dec_cols), cfg['hidden_dim'], 
                            cfg['drop_rate'], cfg['num_layers']).to(dev)
    
    # Load pre-trained weights
    model.load_state_dict(torch.load(f'{model_dir}/model_power.pth', map_location=dev))
    print("✅ Pre-trained model loaded successfully")
    
    # Setup for fine-tuning
    crit = WeightedL1Loss(cfg['future_steps'], dev)
    opt = torch.optim.AdamW(model.parameters(), lr=FINETUNE_CFG['lr'], weight_decay=1e-6)
    sch = ReduceLROnPlateau(opt, 'min', patience=5, factor=.8)
    
    # Fine-tuning loop
    best = 1e9
    wait = 0
    print("⏳ Fine-tuning...")
    
    for ep in range(1, FINETUNE_CFG['epochs'] + 1):
        model.train()
        ft_loss = 0
        
        for xe, xd, yy_power, yy_not_use_power in ft_loader:
            xe, xd, yy_power, yy_not_use_power = xe.to(dev), xd.to(dev), yy_power.to(dev), yy_not_use_power.to(dev)
            opt.zero_grad()
            pred_power, pred_not_use_power = model(xe, xd)
            loss_power = crit(pred_power, yy_power)
            loss_not_use_power = crit(pred_not_use_power, yy_not_use_power)
            loss = FINETUNE_CFG['power_weight'] * loss_power + FINETUNE_CFG['not_use_power_weight'] * loss_not_use_power
            loss.backward()
            opt.step()
            ft_loss += loss.item()
        
        ft_loss /= len(ft_loader)
        sch.step(ft_loss)
        
        if ep % 5 == 0:
            print(f'FT-E{ep:03d} loss{ft_loss:.4f}')
        
        if ft_loss < best:
            best = ft_loss
            wait = 0
            # Save fine-tuned model
            torch.save(model.state_dict(), f'{model_dir}/model_power_finetuned.pth')
        else:
            wait += 1
            if wait >= FINETUNE_CFG['patience']:
                print("ℹ️  Early stopping")
                break
    
    # Update station info
    station_info['finetuned'] = True
    station_info['finetune_data_points'] = len(df_new)
    station_info['finetune_samples'] = len(Xp)
    station_info['finetune_time'] = time.time() - t0
    joblib.dump(station_info, f'{model_dir}/station_info.pkl')
    
    print(f"✅ Fine-tuning completed!")
    print(f"📊 Fine-tuning data points: {len(df_new):,}")
    print(f"🍱 Fine-tuning samples: {len(Xp):,}")
    print(f"⏱️  Fine-tuning time: {time.time()-t0:.1f}s")
    print(f"💾 Fine-tuned model saved to: {model_dir}/model_power_finetuned.pth")
    
    return model_dir

def main():
    parser = argparse.ArgumentParser(description='Fine-tune single station power prediction model')
    parser.add_argument('--model_dir', type=str, required=True, 
                       help='Directory containing the pre-trained model')
    parser.add_argument('--data_file', type=str, required=True,
                       help='Incremental data file for fine-tuning')
    parser.add_argument('--station_id', type=str, required=True,
                       help='Station ID to fine-tune model for')
    
    args = parser.parse_args()
    
    try:
        model_dir = finetune_model(args.model_dir, args.data_file, args.station_id)
        print(f"🎉 Fine-tuning completed successfully! Model updated in: {model_dir}")
    except Exception as e:
        print(f"❌ Fine-tuning failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
