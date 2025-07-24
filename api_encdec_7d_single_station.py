# =========================================================
#  API Service for Single Station Power Prediction
#  Provides REST API for 7-day power forecasting
# =========================================================
import os, warnings, time
import pandas as pd, numpy as np
import torch, torch.nn as nn
import joblib
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import holidays
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Global variables to store loaded models
MODELS = {}
STATION_CONFIGS = {}

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

def make_is_peak(ts):
    h, mi = ts.dt.hour, ts.dt.minute
    return ((h>8)|((h==8)&(mi>=30))) & ((h<17)|((h==17)&(mi<=30)))

def enrich_features(d: pd.DataFrame):
    """Feature engineering function"""
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

def load_station_model(station_id):
    """Load model components for a specific station"""
    model_dir = f"models/station_{station_id}"
    
    if not os.path.exists(model_dir):
        raise ValueError(f"Model not found for station {station_id}")
    
    # Load configuration and components
    cfg = joblib.load(f'{model_dir}/config.pkl')
    enc_cols = joblib.load(f'{model_dir}/enc_cols.pkl')
    dec_cols = joblib.load(f'{model_dir}/dec_cols.pkl')
    sc_e = joblib.load(f'{model_dir}/scaler_enc.pkl')
    sc_d = joblib.load(f'{model_dir}/scaler_dec.pkl')
    sc_y_power = joblib.load(f'{model_dir}/scaler_y_power.pkl')
    sc_y_not_use_power = joblib.load(f'{model_dir}/scaler_y_not_use_power.pkl')
    station_info = joblib.load(f'{model_dir}/station_info.pkl')
    
    # Load model
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EncDecPowerModel(len(enc_cols), len(dec_cols), cfg['hidden_dim'], 
                            cfg['drop_rate'], cfg['num_layers']).to(dev)
    
    # Try to load fine-tuned model first, fallback to original
    model_path = f'{model_dir}/model_power_finetuned.pth'
    if not os.path.exists(model_path):
        model_path = f'{model_dir}/model_power.pth'
    
    model.load_state_dict(torch.load(model_path, map_location=dev))
    model.eval()
    
    return {
        'model': model,
        'config': cfg,
        'enc_cols': enc_cols,
        'dec_cols': dec_cols,
        'scaler_enc': sc_e,
        'scaler_dec': sc_d,
        'scaler_y_power': sc_y_power,
        'scaler_y_not_use_power': sc_y_not_use_power,
        'station_info': station_info,
        'device': dev
    }

def prepare_input_data(data, station_id):
    """Prepare input data for prediction"""
    # Convert to DataFrame if it's a list of dictionaries
    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data.copy()
    
    # Ensure required columns
    if 'ts' not in df.columns:
        raise ValueError("Missing 'ts' field in input data")
    if 'total_active_power' not in df.columns:
        raise ValueError("Missing 'total_active_power' field in input data")
    
    # Parse timestamp
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.sort_values('ts')
    
    # Add station_ref_id if not present
    if 'station_ref_id' not in df.columns:
        df['station_ref_id'] = station_id
    
    # Create not_use_power if not exists
    if 'not_use_power' not in df.columns:
        df['not_use_power'] = df['total_active_power'] * 0.7
    
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
    
    return df

def generate_future_timestamps(last_timestamp, future_steps=672):
    """Generate future timestamps for prediction (15-min intervals)"""
    timestamps = []
    current_time = pd.to_datetime(last_timestamp)
    
    for i in range(1, future_steps + 1):
        future_time = current_time + timedelta(minutes=15 * i)
        timestamps.append(future_time)
    
    return timestamps

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'loaded_stations': list(MODELS.keys())
    })

@app.route('/load_model/<station_id>', methods=['POST'])
def load_model_endpoint(station_id):
    """Load model for a specific station"""
    try:
        model_components = load_station_model(station_id)
        MODELS[station_id] = model_components
        STATION_CONFIGS[station_id] = model_components['config']
        
        return jsonify({
            'status': 'success',
            'message': f'Model loaded for station {station_id}',
            'station_info': model_components['station_info']
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/predict/<station_id>', methods=['POST'])
def predict_power(station_id):
    """Predict 7-day power consumption for a station"""
    try:
        # Check if model is loaded
        if station_id not in MODELS:
            try:
                model_components = load_station_model(station_id)
                MODELS[station_id] = model_components
                STATION_CONFIGS[station_id] = model_components['config']
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': f'Failed to load model for station {station_id}: {str(e)}'
                }), 400
        
        # Get request data
        request_data = request.get_json()
        if not request_data or 'data' not in request_data:
            return jsonify({
                'status': 'error',
                'message': 'Missing input data'
            }), 400
        
        input_data = request_data['data']
        
        # Load model components
        model_components = MODELS[station_id]
        model = model_components['model']
        cfg = model_components['config']
        enc_cols = model_components['enc_cols']
        dec_cols = model_components['dec_cols']
        sc_e = model_components['scaler_enc']
        sc_d = model_components['scaler_dec']
        sc_y_power = model_components['scaler_y_power']
        sc_y_not_use_power = model_components['scaler_y_not_use_power']
        dev = model_components['device']
        
        # Prepare input data
        df = prepare_input_data(input_data, station_id)
        
        # Check if we have enough historical data
        if len(df) < cfg['past_steps']:
            return jsonify({
                'status': 'error',
                'message': f'Insufficient historical data. Need {cfg["past_steps"]} points, got {len(df)}'
            }), 400
        
        # Use the most recent data for prediction
        recent_data = df.tail(cfg['past_steps'] + cfg['future_steps'])
        
        # Handle missing columns
        missing_cols = []
        for col in enc_cols + dec_cols:
            if col not in recent_data.columns:
                missing_cols.append(col)
                recent_data[col] = 0
        
        if missing_cols:
            print(f"Warning: Missing columns filled with zeros: {missing_cols}")
        
        # Prepare model input
        xe = sc_e.transform(recent_data[enc_cols])[:cfg['past_steps']].astype(np.float32)
        
        # For decoder input, we need to create future features
        # Use the last known values and extend them
        last_values = recent_data[dec_cols].iloc[-1:].values
        xd_future = np.tile(last_values, (cfg['future_steps'], 1)).astype(np.float32)
        xd = sc_d.transform(xd_future)
        
        # Convert to tensors
        xe_tensor = torch.from_numpy(xe).unsqueeze(0).to(dev)
        xd_tensor = torch.from_numpy(xd).unsqueeze(0).to(dev)
        
        # Make prediction
        with torch.no_grad():
            pred_power_scaled, pred_not_use_power_scaled = model(xe_tensor, xd_tensor)
            pred_power_scaled = pred_power_scaled.cpu().numpy().flatten()
            pred_not_use_power_scaled = pred_not_use_power_scaled.cpu().numpy().flatten()
        
        # Inverse transform predictions
        pred_power = sc_y_power.inverse_transform(pred_power_scaled.reshape(-1, 1)).flatten()
        pred_not_use_power = sc_y_not_use_power.inverse_transform(pred_not_use_power_scaled.reshape(-1, 1)).flatten()
        
        # Ensure non-negative predictions
        pred_power = np.maximum(pred_power, 0)
        pred_not_use_power = np.maximum(pred_not_use_power, 0)
        
        # Generate future timestamps
        last_timestamp = df['energy_date'].iloc[-1]
        future_timestamps = generate_future_timestamps(last_timestamp, cfg['future_steps'])
        
        # Prepare response
        predictions = []
        for i, (ts, power, not_use_power) in enumerate(zip(future_timestamps, pred_power, pred_not_use_power)):
            predictions.append({
                'timestamp': ts.isoformat(),
                'total_active_power': float(power),
                'not_use_power': float(not_use_power),
                'step': i + 1,
                'day': (i // 96) + 1,
                'hour_of_day': ts.hour,
                'minute_of_hour': ts.minute
            })
        
        return jsonify({
            'status': 'success',
            'station_id': station_id,
            'prediction_start': future_timestamps[0].isoformat(),
            'prediction_end': future_timestamps[-1].isoformat(),
            'total_points': len(predictions),
            'predictions': predictions,
            'metadata': {
                'model_type': 'EncDec_7d_Power',
                'prediction_horizon_days': 7,
                'interval_minutes': 15,
                'input_data_points': len(df),
                'missing_columns_filled': missing_cols
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/stations', methods=['GET'])
def list_stations():
    """List available station models"""
    try:
        models_dir = "models"
        if not os.path.exists(models_dir):
            return jsonify({
                'status': 'success',
                'stations': []
            })
        
        stations = []
        for item in os.listdir(models_dir):
            if item.startswith('station_') and os.path.isdir(os.path.join(models_dir, item)):
                station_id = item.replace('station_', '')
                station_info_path = os.path.join(models_dir, item, 'station_info.pkl')
                
                if os.path.exists(station_info_path):
                    try:
                        station_info = joblib.load(station_info_path)
                        stations.append({
                            'station_id': station_id,
                            'loaded': station_id in MODELS,
                            'info': station_info
                        })
                    except:
                        stations.append({
                            'station_id': station_id,
                            'loaded': station_id in MODELS,
                            'info': None
                        })
        
        return jsonify({
            'status': 'success',
            'stations': stations
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Power Prediction API Server...")
    print("📊 Available endpoints:")
    print("  GET  /health - Health check")
    print("  GET  /stations - List available stations")
    print("  POST /load_model/<station_id> - Load model for station")
    print("  POST /predict/<station_id> - Predict power for station")
    print("🌐 Server starting on http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
