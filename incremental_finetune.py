#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Incrementally fine-tune a pre-trained power prediction model
"""
import os
import time
import argparse
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_percentage_error
import warnings
import holidays
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings("ignore")

# Define the model architecture to match exactly what's in train_encdec_7d_single_station.py
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

def load_model_components(model_dir):
    """
    Load all model components from a saved model directory.
    """
    print(f"Loading model components from {model_dir}...")
    
    # Load scalers
    sc_e = joblib.load(f'{model_dir}/scaler_enc.pkl')
    sc_d = joblib.load(f'{model_dir}/scaler_dec.pkl')
    sc_y_power = joblib.load(f'{model_dir}/scaler_y_power.pkl')
    sc_y_not_use_power = joblib.load(f'{model_dir}/scaler_y_not_use_power.pkl')
    
    # Load feature lists
    ENC_COLS = joblib.load(f'{model_dir}/enc_cols.pkl')
    DEC_COLS = joblib.load(f'{model_dir}/dec_cols.pkl')
    
    # Load config
    config = joblib.load(f'{model_dir}/config.pkl')
    
    # Create model with the same architecture
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EncDecPowerModel(
        len(ENC_COLS), 
        len(DEC_COLS), 
        config['hidden_dim'], 
        config['drop_rate'], 
        config['num_layers']
    ).to(dev)
    
    # Load model weights
    state_dict = torch.load(f'{model_dir}/model_power.pth', map_location=dev)
    
    # Check if the model architecture matches the saved model
    print("Loading model weights...")
    model.load_state_dict(state_dict)
    
    return {
        'model': model,
        'sc_e': sc_e,
        'sc_d': sc_d,
        'sc_y_power': sc_y_power,
        'sc_y_not_use_power': sc_y_not_use_power,
        'ENC_COLS': ENC_COLS,
        'DEC_COLS': DEC_COLS,
        'config': config,
        'device': dev
    }

def load_best_params(station_id):
    """
    Load the best hyperparameters for a given station from the saved file.
    If the file doesn't exist, return None.
    
    Args:
        station_id: Station ID to read parameters for
    
    Returns:
        best_params: Dictionary of best parameters or None if file doesn't exist
    """
    study_dir = "hyperparameter_tuning_results"
    file_path = f"{study_dir}/best_params_station_{station_id}.pkl"
    
    if not os.path.exists(file_path):
        print(f"No best parameters file found at {file_path}. Will use default parameters.")
        return None
    
    best_params = joblib.load(file_path)
    
    print("\nLoaded best hyperparameters for station", station_id)
    print("="*50)
    
    # Print MAPE if available
    if "best_mape" in best_params:
        print(f"Best MAPE: {best_params['best_mape']:.2%}")
        # Remove MAPE from parameters
        best_mape = best_params.pop("best_mape")
        
    # Print parameters
    for key, value in best_params.items():
        print(f"{key}: {value}")
    
    return best_params

def get_default_params():
    """
    Return default optimized parameters.
    """
    return {
        'hidden_dim': 512,  # Match the original model's hidden dimension
        'num_layers': 1,    # Match the original model's number of layers
        'drop_rate': 0.4,   # Match the original model's dropout rate
        'batch_size': 256,  # Match the original model's batch size
        'lr': 0.0001,       # Match the original model's learning rate
        'top_k': 90,
        'use_stl': True,
        'power_weight': 1.0,
        'not_use_power_weight': 0.7,
    }

def create_features_like_training_script(df):
    """
    Create features exactly like in train_encdec_7d_single_station.py
    """
    import holidays
    from statsmodels.tsa.seasonal import STL
    
    df = df.copy()
    
    # Rename ts to energy_date for compatibility
    df = df.rename(columns={'ts': 'energy_date'})
    
    # Parse timestamp
    df['energy_date'] = pd.to_datetime(df['energy_date'])
    df = df.sort_values(['station_ref_id', 'energy_date'])
    
    # Create not_use_power if not exists
    if 'not_use_power' not in df.columns:
        df['not_use_power'] = df['total_active_power'] * 0.7
        print("WARNING: not_use_power field created as 70% of total_active_power")
    
    # Clean power data
    df['total_active_power'] = df['total_active_power'].clip(lower=0)
    df['not_use_power'] = df['not_use_power'].clip(lower=0)
    
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
    # Time/Weather/Holiday Features (exactly like training script)
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
    # Power-based Features (exactly like training script)
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
    df['power_ratio'] = df['power_ratio'].clip(0, 1)
    
    # =========================================================
    # STL Decomposition + Quantile Normalization (exactly like training script)
    # =========================================================
    def add_advanced_features(g):
        try:
            if len(g)>=96*14:
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
        except:
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
    # Data Cleaning (exactly like training script)
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
    
    return df

def make_dataset(df, past_steps, future_steps, ENC_COLS, DEC_COLS, sc_e, sc_d, sc_y_power, sc_y_not_use_power):
    """
    Create dataset from dataframe using the saved scalers and feature columns.
    Exactly like the training script.
    """
    # Apply the same feature engineering as the training script
    df = create_features_like_training_script(df)
    
    print("Validating data quality...")
    for col in ENC_COLS + DEC_COLS + ['total_active_power', 'not_use_power']:
        if col in df.columns:
            if df[col].isnull().any():
                print(f"WARNING: {col} has {df[col].isnull().sum()} null values, filling with median")
                df[col] = df[col].fillna(df[col].median())
            if np.isinf(df[col]).any():
                print(f"WARNING: {col} has infinite values, replacing with boundary values")
                df[col] = df[col].replace([np.inf, -np.inf], 
                                        [df[col].quantile(0.99), df[col].quantile(0.01)])
    
    # Create feature matrices with available columns, fill missing ones with zeros
    enc_feature_matrix = np.zeros((len(df), len(ENC_COLS)), dtype=np.float32)
    dec_feature_matrix = np.zeros((len(df), len(DEC_COLS)), dtype=np.float32)
    
    # Fill available encoder features
    for i, col in enumerate(ENC_COLS):
        if col in df.columns:
            enc_feature_matrix[:, i] = df[col].values.astype(np.float32)
        else:
            print(f"Missing encoder feature: {col}, filled with zeros")
    
    # Fill available decoder features
    for i, col in enumerate(DEC_COLS):
        if col in df.columns:
            dec_feature_matrix[:, i] = df[col].values.astype(np.float32)
        else:
            print(f"Missing decoder feature: {col}, filled with zeros")
    
    # Get target values
    power_values = df['total_active_power'].values.astype(np.float32)
    not_use_power_values = df['not_use_power'].values.astype(np.float32)
    
    # Apply the saved scalers
    enc_features_scaled = sc_e.transform(enc_feature_matrix)
    dec_features_scaled = sc_d.transform(dec_feature_matrix)
    power_scaled = sc_y_power.transform(power_values.reshape(-1, 1)).flatten()
    not_use_power_scaled = sc_y_not_use_power.transform(not_use_power_values.reshape(-1, 1)).flatten()
    
    # Create sequences
    Xp = []  # past features
    Xf = []  # future features
    Y_power = []  # power targets
    Y_not_use_power = []  # not_use_power targets
    
    for i in range(len(df) - past_steps - future_steps + 1):
        # Past window (encoder features)
        past_window = enc_features_scaled[i:i+past_steps]
        # Future window (decoder features)
        future_window = dec_features_scaled[i+past_steps:i+past_steps+future_steps]
        
        # Future targets
        power_targets = power_scaled[i+past_steps:i+past_steps+future_steps]
        not_use_power_targets = not_use_power_scaled[i+past_steps:i+past_steps+future_steps]
        
        Xp.append(past_window)
        Xf.append(future_window)
        Y_power.append(power_targets)
        Y_not_use_power.append(not_use_power_targets)
    
    print(f"Created dataset with {len(Xp)} samples")
    print(f"Encoder input shape: {np.array(Xp).shape}")
    print(f"Decoder input shape: {np.array(Xf).shape}")
    
    return (
        np.array(Xp, dtype=np.float32), 
        np.array(Xf, dtype=np.float32), 
        np.array(Y_power, dtype=np.float32), 
        np.array(Y_not_use_power, dtype=np.float32),
        None, None, None, None  # Placeholder for other returns
    )

def incremental_finetune(model_dir, station_id=None, data_file='merged_station_test.csv', 
                         epochs=50, lr=1e-5, focus_days=None):
    """
    Incrementally fine-tune a pre-trained model with a focus on specific days.
    
    Args:
        model_dir: Directory containing the pre-trained model
        station_id: Station ID to fine-tune for
        data_file: Input data file
        epochs: Number of fine-tuning epochs
        lr: Learning rate for fine-tuning
        focus_days: List of days to focus on (1-7), or None for all days
    """
    t0 = time.time()
    
    # Load model components
    components = load_model_components(model_dir)
    model = components['model']
    sc_e = components['sc_e']
    sc_d = components['sc_d']
    sc_y_power = components['sc_y_power']
    sc_y_not_use_power = components['sc_y_not_use_power']
    ENC_COLS = components['ENC_COLS']
    DEC_COLS = components['DEC_COLS']
    config = components['config']
    dev = components['device']
    
    # Use the original model directory to overwrite (directly modify the original model)
    output_dir = model_dir
    
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Will overwrite the original model in: {output_dir}")
    
    # Load and prepare data
    print(f"Loading data from {data_file}...")
    
    # Read and process data
    df = pd.read_csv(data_file, dtype={'station_ref_id': str})
    df = df.sort_values(['station_ref_id', 'ts'])
    
    # Filter for station
    if station_id:
        df = df[df['station_ref_id'] == station_id].copy()
        if len(df) == 0:
            raise ValueError(f"No data found for station {station_id}")
        print(f"Fine-tuning model for station: {station_id}")
    else:
        # Use first station if none specified
        station_id = df['station_ref_id'].iloc[0]
        df = df[df['station_ref_id'] == station_id].copy()
        print(f"Fine-tuning model for station: {station_id} (first station)")
    
    # Try to load best parameters for this station
    best_params = load_best_params(station_id)
    
    # If best parameters exist, update config with them
    if best_params:
        print("Using best parameters from tuning results")
        for key, value in best_params.items():
            if key in config:
                config[key] = value
    else:
        # Use default parameters
        print("Using default parameters")
        default_params = get_default_params()
        for key, value in default_params.items():
            if key in config:
                config[key] = value
    
    # Create dataset
    Xp, Xf, Y_power, Y_not_use_power, _, _, _, _ = make_dataset(
        df, config['past_steps'], config['future_steps'], 
        ENC_COLS, DEC_COLS, sc_e, sc_d, sc_y_power, sc_y_not_use_power
    )
    
    print(f"Samples: {len(Xp)}")
    
    if len(Xp) == 0:
        raise ValueError("No samples generated. Check data length and parameters.")
    
    # Create train/validation split
    spl = int(.8 * len(Xp))
    
    # Convert numpy arrays to torch tensors with explicit float32 type
    tr_ds = TensorDataset(
        torch.tensor(Xp[:spl], dtype=torch.float32), 
        torch.tensor(Xf[:spl], dtype=torch.float32),
        torch.tensor(Y_power[:spl], dtype=torch.float32), 
        torch.tensor(Y_not_use_power[:spl], dtype=torch.float32)
    )
    va_ds = TensorDataset(
        torch.tensor(Xp[spl:], dtype=torch.float32), 
        torch.tensor(Xf[spl:], dtype=torch.float32),
        torch.tensor(Y_power[spl:], dtype=torch.float32), 
        torch.tensor(Y_not_use_power[spl:], dtype=torch.float32)
    )
    
    tr_loader = DataLoader(tr_ds, batch_size=config['batch_size'], shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=config['batch_size'], shuffle=False)
    
    # Create custom loss function with focus on specific days
    class FocusedWeightedL1Loss(nn.Module):
        def __init__(self, fut, device, focus_days=None):
            super().__init__()
            # Base weights - higher for days 3-4
            w = np.concatenate([
                np.ones(96*2),           # Day1-2
                np.ones(96)*1.3,         # Day3
                np.ones(96)*1.5,         # Day4
                np.ones(96*3)*1.2        # Day5-7
            ])
            
            # Apply additional focus on specific days
            if focus_days:
                for day in focus_days:
                    if 1 <= day <= 7:
                        start_idx = (day - 1) * 96
                        end_idx = day * 96
                        w[start_idx:end_idx] *= 2.0  # Double the weight for focus days
            
            self.register_buffer('w', torch.tensor(w, dtype=torch.float32, device=device))
            
        def forward(self, pred, target):
            return torch.mean(self.w * torch.abs(pred - target))
    
    # Set up fine-tuning
    model.train()
    crit = FocusedWeightedL1Loss(config['future_steps'], dev, focus_days)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    sch = ReduceLROnPlateau(opt, 'min', patience=5, factor=.5)
    
    best = 1e9
    wait = 0
    patience = 15
    print("Fine-tuning...")
    
    for ep in range(1, epochs + 1):
        model.train()
        tr = 0
        for xe, xd, yy_power, yy_not_use_power in tr_loader:
            # Ensure correct data type and move to device
            xe = xe.to(dev).float()
            xd = xd.to(dev).float()
            yy_power = yy_power.to(dev).float()
            yy_not_use_power = yy_not_use_power.to(dev).float()
            
            opt.zero_grad()
            pred_power, pred_not_use_power = model(xe, xd)
            loss_power = crit(pred_power, yy_power)
            loss_not_use_power = crit(pred_not_use_power, yy_not_use_power)
            loss = config['power_weight'] * loss_power + config['not_use_power_weight'] * loss_not_use_power
            loss.backward()
            opt.step()
            tr += loss.item()
        tr /= len(tr_loader)
        
        model.eval()
        va = 0
        with torch.no_grad():
            for xe, xd, yy_power, yy_not_use_power in va_loader:
                # Ensure correct data type and move to device
                xe = xe.to(dev).float()
                xd = xd.to(dev).float()
                yy_power = yy_power.to(dev).float()
                yy_not_use_power = yy_not_use_power.to(dev).float()
                
                pred_power, pred_not_use_power = model(xe, xd)
                loss_power = crit(pred_power, yy_power)
                loss_not_use_power = crit(pred_not_use_power, yy_not_use_power)
                va += (config['power_weight'] * loss_power + config['not_use_power_weight'] * loss_not_use_power).item()
        va /= len(va_loader)
        sch.step(va)
        
        if ep % 5 == 0:
            print(f'E{ep:03d} tr{tr:.4f} va{va:.4f}')
        
        if va < best:
            best = va
            wait = 0
            torch.save(model.state_dict(), f'{output_dir}/model_power.pth')
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping")
                break
    
    # Evaluate MAPE for each day
    print("Evaluating MAPE for last 7 days prediction...")
    
    # Load the best model
    model.load_state_dict(torch.load(f'{output_dir}/model_power.pth'))
    model.eval()
    
    all_power_preds = []
    all_not_use_power_preds = []
    all_power_targets = []
    all_not_use_power_targets = []
    
    with torch.no_grad():
        for xe, xd, yy_power, yy_not_use_power in va_loader:
            # Ensure correct data type and move to device
            xe = xe.to(dev).float()
            xd = xd.to(dev).float()
            
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
    
    # Save model components
    joblib.dump(sc_e, f'{output_dir}/scaler_enc.pkl')
    joblib.dump(sc_d, f'{output_dir}/scaler_dec.pkl')
    joblib.dump(sc_y_power, f'{output_dir}/scaler_y_power.pkl')
    joblib.dump(sc_y_not_use_power, f'{output_dir}/scaler_y_not_use_power.pkl')
    joblib.dump(ENC_COLS, f'{output_dir}/enc_cols.pkl')
    joblib.dump(DEC_COLS, f'{output_dir}/dec_cols.pkl')
    joblib.dump(config, f'{output_dir}/config.pkl')
    
    # Save station info
    station_info = {
        'station_id': station_id,
        'fine_tuned_from': model_dir,
        'focus_days': focus_days,
        'fine_tuning_epochs': epochs,
        'fine_tuning_lr': lr
    }
    joblib.dump(station_info, f'{output_dir}/station_info.pkl')
    
    print(f"Fine-tuned model saved to {output_dir}/")
    print(f"Total time: {time.time()-t0:.1f}s")
    
    return output_dir, overall_power_mape

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Incrementally fine-tune a pre-trained power prediction model')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing the pre-trained model')
    parser.add_argument('--station_id', type=str,
                       help='Station ID to fine-tune for')
    parser.add_argument('--data_file', type=str, default='merged_station_test.csv', 
                       help='Input data file')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of fine-tuning epochs')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate for fine-tuning')
    parser.add_argument('--focus_days', type=int, nargs='+',
                       help='Days to focus on (1-7)')
    
    args = parser.parse_args()
    
    try:
        output_dir, mape = incremental_finetune(
            args.model_dir,
            args.station_id,
            args.data_file,
            args.epochs,
            args.lr,
            args.focus_days
        )
        print(f"Fine-tuning completed successfully!")
        print(f"Final MAPE: {mape:.2%}")
    except Exception as e:
        print(f"Fine-tuning failed: {str(e)}")
        raise
