# train_the_best_model.py
import os, joblib, warnings
import pandas as pd
import numpy as np
import holidays
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.makedirs('output', exist_ok=True) 

# === 0. 超参 (已被验证为最高效的配置) ===
CFG = dict(
    past_steps   = 96 * 5,
    future_steps = 96,
    lstm_units   = [256, 128],
    drop_rate    = 0.4,
    batch_size   = 512,
    epochs       = 150,
    patience     = 20,
    lr           = 8e-4
)

# === 1. 读数 ===
# 采用最有效的方式：不预先处理'NULL'，让后续流程通过填充来处理
df = pd.read_csv('load.csv', parse_dates=['energy_date'])
df = df.sort_values(['station_ref_id', 'energy_date'])

# === 2. 时间特征 & 节假日 ===
cn_holidays = holidays.country_holidays('CN')

def add_time_feat(df):
    df = df.copy()
    df['hour'] = df['energy_date'].dt.hour
    df['weekday'] = df['energy_date'].dt.weekday
    df['day'] = df['energy_date'].dt.day
    df['month'] = df['energy_date'].dt.month

    # 确保目标列在分组计算前是数字，'NULL'等字符串会被转为NaN
    df['load_discharge_delta'] = pd.to_numeric(df['load_discharge_delta'], errors='coerce')
    # 使用ffill填充因转换产生的NaN，这保留了数据点
    df['load_discharge_delta'] = df.groupby('station_ref_id')['load_discharge_delta'].ffill()


    for lag in [1, 4, 96]:
        df[f'load_lag{lag}'] = df.groupby('station_ref_id')['load_discharge_delta'].shift(lag)
    
    for win in [4, 12, 96]:
        grouped_load = df.groupby('station_ref_id')['load_discharge_delta']
        df[f'load_ma{win}'] = grouped_load.rolling(win, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'load_std{win}'] = grouped_load.rolling(win, min_periods=1).std().reset_index(level=0, drop=True)

    df['is_holiday'] = df['energy_date'].isin(cn_holidays).astype(int)
    df['is_month_start'] = (df['day'] <= 3).astype(int)
    df['is_month_end'] = (df['day'] >= 28).astype(int)

    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['sin_wday'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['cos_wday'] = np.cos(2 * np.pi * df['weekday'] / 7)
    return df

df = add_time_feat(df)

# === 3. 数据清洗 & NaN 处理 ===
df = df.set_index('energy_date')

for sid in df['station_ref_id'].unique():
    station_mask = df['station_ref_id'] == sid
    mean, std = df.loc[station_mask, 'load_discharge_delta'].mean(), df.loc[station_mask, 'load_discharge_delta'].std()
    low, high = mean - 3 * std, mean + 3 * std
    df.loc[station_mask, 'load_discharge_delta'] = df.loc[station_mask, 'load_discharge_delta'].clip(lower=low, upper=high)

# 此处fillna主要处理因特征工程(lag/rolling)在开头产生的NaN
df = df.fillna(method='ffill').dropna()
df = df.reset_index()

# === 4. station 编码 ===
le = LabelEncoder()
df['station_enc'] = le.fit_transform(df['station_ref_id'])

# === 5. 构造数据集 ===
def make_dataset(data, past_steps, future_steps):
    data = data.sort_values(['station_enc', 'energy_date']).set_index('energy_date')
    feature_cols = [
        'temp', 'code', 'humidity', 'windSpeed', 'cloud', 'is_work', 'is_peak',
        'load_lag1', 'load_lag4', 'load_lag96', 'load_ma4', 'load_ma12', 'load_ma96',
        'load_std4', 'load_std12', 'load_std96', 'is_holiday', 'is_month_start', 'is_month_end',
        'sin_hour', 'cos_hour', 'sin_wday', 'cos_wday', 'station_enc'
    ]
    target_col = 'load_discharge_delta'
    scaler_x, scaler_y = StandardScaler(), StandardScaler()
    
    # 确保所有特征列都是数值
    for col in feature_cols + [target_col]:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.fillna(method='ffill').dropna()
    
    if data.empty: return np.array([]), np.array([]), None, None

    X_scaled = scaler_x.fit_transform(data[feature_cols])
    y_scaled = scaler_y.fit_transform(data[[target_col]])
    
    scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=data.index)
    scaled_df[target_col] = y_scaled

    Xs, ys = [], []
    for sid_enc in scaled_df['station_enc'].unique():
        station_df = scaled_df[scaled_df['station_enc'] == sid_enc]
        X_station = station_df[feature_cols].values
        y_station = station_df[[target_col]].values
        for i in range(len(X_station) - past_steps - future_steps + 1):
            Xs.append(X_station[i:i + past_steps])
            ys.append(y_station[i + past_steps:i + past_steps + future_steps, 0])

    return np.array(Xs), np.array(ys), scaler_x, scaler_y

# === 6. 训练全局模型 ===
print('=== Training The High-Performing Global Model ===')
if df.empty:
    print("数据为空，无法训练。")
    exit()

X, y, scaler_x_fitted, scaler_y_fitted = make_dataset(df, CFG['past_steps'], CFG['future_steps'])

if len(X) == 0:
    print("制作滑窗样本后数据为空，请检查数据量。")
    exit()

split = int(0.8 * len(X))
X_train, X_val, y_train, y_val = X[:split], X[split:], y[:split], y[split:]

print(f"Total samples: {len(X)}")
print(f"Train samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

model = Sequential([
    LSTM(CFG['lstm_units'][0], return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(CFG['drop_rate']),
    LSTM(CFG['lstm_units'][1]),
    Dropout(CFG['drop_rate']),
    Dense(CFG['future_steps'])
])

model.compile(optimizer=tf.keras.optimizers.Adam(CFG['lr']), loss='mae')
model.summary()

cb = [
    EarlyStopping(monitor='val_loss', patience=CFG['patience'], restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
]

history = model.fit(
    X_train, y_train, validation_data=(X_val, y_val),
    epochs=CFG['epochs'], batch_size=CFG['batch_size'], callbacks=cb, verbose=1
)

# === 7. 评估 & 可视化 ===
all_preds, all_trues = [], []
for sid, g in df.groupby('station_ref_id'):
    if len(g) < CFG['past_steps'] + CFG['future_steps']:
        print(f"Skipping {sid}, not enough data for a test sample.")
        continue
    
    print(f'--- Evaluating {sid} ---')
    
    X_test_unscaled = g.tail(CFG['past_steps'] + CFG['future_steps'])
    X_test_scaled = scaler_x_fitted.transform(X_test_unscaled[scaler_x_fitted.feature_names_in_])
    X_test = X_test_scaled[:-CFG['future_steps']].reshape(1, CFG['past_steps'], -1)
    
    pred_scaled = model.predict(X_test, verbose=0)[0]
    pred = scaler_y_fitted.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    true = X_test_unscaled.tail(CFG['future_steps'])['load_discharge_delta'].values
    
    all_preds.append(pred)
    all_trues.append(true)

    mape = mean_absolute_percentage_error(true, pred) * 100
    rmse = np.sqrt(mean_squared_error(true, pred))
    print(f'{sid}  MAPE={mape:.2f}%   RMSE={rmse:.2f}')

    idx = g['energy_date'].iloc[-CFG['future_steps']:].reset_index(drop=True)
    plt.figure(figsize=(15, 4))
    plt.plot(idx, true, label='Real', marker='.', linestyle='-')
    plt.plot(idx, pred, label='Pred', marker='.', linestyle='--')
    plt.title(f'Prediction for {sid}  |  MAPE={mape:.2f}%  RMSE={rmse:.2f}')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f'output/test_{sid}.png')
    plt.close()

if all_trues:
    overall_mape = mean_absolute_percentage_error(np.concatenate(all_trues), np.concatenate(all_preds)) * 100
    overall_rmse = np.sqrt(mean_squared_error(np.concatenate(all_trues), np.concatenate(all_preds)))
    print(f'\n=== Overall Performance ===')
    print(f'Overall MAPE: {overall_mape:.2f}%')
    print(f'Overall RMSE: {overall_rmse:.2f}')

# === 8. 保存 ===
print('\nSaving global model and scalers...')
model.save('output/model_global.h5')
joblib.dump(scaler_x_fitted, 'output/scaler_x_global.pkl')
joblib.dump(scaler_y_fitted, 'output/scaler_y_global.pkl')
joblib.dump(le, 'output/label_encoder.pkl')
print('All assets saved to output/')