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

# === 0. 优化超参 ===
CFG = dict(
    past_steps   = 96 * 5,  # 5天历史数据 (保持有效时间窗口)
    future_steps = 96,       # 预测未来96个时间点
    lstm_units   = [256, 128],  # 优化LSTM单元数量
    drop_rate    = 0.4,      # 适度Dropout
    batch_size   = 256,      # 优化batch_size
    epochs       = 150,      # 适度训练轮次
    patience     = 15,
    lr           = 5e-4     # 稍高学习率
)

# === 1. 读数 ===
# 注意：请确保CSV文件名与您的文件名一致，此处使用 'loaddata.csv'
df = pd.read_csv('loaddata.csv', parse_dates=['energy_date'])
df = df.sort_values(['station_ref_id', 'energy_date'])

# === 2. 核心特征工程 ===
cn_holidays = holidays.country_holidays('CN')

def add_time_feat(df):
    df = df.copy()
    # 核心时间特征
    df['hour'] = df['energy_date'].dt.hour
    df['weekday'] = df['energy_date'].dt.weekday
    df['day'] = df['energy_date'].dt.day
    df['month'] = df['energy_date'].dt.month

    # 处理目标列
    df['load_discharge_delta'] = pd.to_numeric(df['load_discharge_delta'], errors='coerce')
    df['load_discharge_delta'] = df.groupby('station_ref_id')['load_discharge_delta'].ffill()

    # 高效滞后特征 - 仅保留最有效的滞后
    lags = [1, 4, 24, 96]  # 小时、4小时、天、4天滞后
    for lag in lags:
        df[f'load_lag{lag}'] = df.groupby('station_ref_id')['load_discharge_delta'].shift(lag)

    # 高效统计特征
    windows = [4, 24, 96]  # 4小时、天、4天窗口
    for win in windows:
        grouped_load = df.groupby('station_ref_id')['load_discharge_delta']
        df[f'load_ma{win}'] = grouped_load.rolling(win, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'load_std{win}'] = grouped_load.rolling(win, min_periods=1).std().reset_index(level=0, drop=True)

    # 节假日特征
    df['is_holiday'] = df['energy_date'].isin(cn_holidays).astype(int)

    # 高效周期性特征
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['sin_wday'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['cos_wday'] = np.cos(2 * np.pi * df['weekday'] / 7)

    # 温度特征增强
    df['temp_squared'] = df['temp'] ** 2  # 非线性温度效应

    return df

df = add_time_feat(df)

# === 3. 数据清洗 & NaN 处理 ===
df = df.set_index('energy_date')

# 仅对目标列进行异常值处理
for sid in df['station_ref_id'].unique():
    station_mask = df['station_ref_id'] == sid
    col_data = df.loc[station_mask, 'load_discharge_delta']
    mean, std = col_data.mean(), col_data.std()
    if std > 0:  # 避免除零
        low, high = mean - 3 * std, mean + 3 * std
        df.loc[station_mask, 'load_discharge_delta'] = col_data.clip(lower=low, upper=high)

df = df.fillna(method='ffill').dropna()
df = df.reset_index()

# === 4. station 编码 ===
le = LabelEncoder()
df['station_enc'] = le.fit_transform(df['station_ref_id'])

# === 5. 构造数据集 ===
def make_dataset(data, past_steps, future_steps):
    data = data.sort_values(['station_enc', 'energy_date']).set_index('energy_date')

    # 核心特征集 - 仅保留高效特征
    feature_cols = [
        'temp', 'temp_squared', 'humidity', 'windSpeed',
        'load_lag1', 'load_lag4', 'load_lag24', 'load_lag96',
        'load_ma4', 'load_ma24', 'load_ma96',
        'load_std4', 'load_std24', 'load_std96',
        'is_holiday', 'sin_hour', 'cos_hour', 'sin_wday', 'cos_wday', 'station_enc'
    ]

    target_col = 'load_discharge_delta'
    scaler_x, scaler_y = StandardScaler(), StandardScaler()

    # 确保所有特征列都是数值
    for col in feature_cols + [target_col]:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data = data.fillna(method='ffill').dropna()

    if data.empty:
        return np.array([]), np.array([]), None, None

    X_scaled = scaler_x.fit_transform(data[feature_cols])
    y_scaled = scaler_y.fit_transform(data[[target_col]])

    scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=data.index)
    scaled_df[target_col] = y_scaled

    Xs, ys = [], []
    for sid_enc in scaled_df['station_enc'].unique():
        station_df = scaled_df[scaled_df['station_enc'] == sid_enc]
        # 确保每个站点有足够数据
        if len(station_df) < past_steps + future_steps:
            continue

        X_station = station_df[feature_cols].values
        y_station = station_df[[target_col]].values

        for i in range(len(X_station) - past_steps - future_steps + 1):
            Xs.append(X_station[i:i + past_steps])
            ys.append(y_station[i + past_steps:i + past_steps + future_steps, 0])

    return np.array(Xs), np.array(ys), scaler_x, scaler_y

# === 6. 训练优化模型 ===
print('=== Training Optimized Model ===')
X, y, scaler_x_fitted, scaler_y_fitted = make_dataset(df, CFG['past_steps'], CFG['future_steps'])

if len(X) == 0:
    print("制作滑窗样本后数据为空，请检查数据量。")
    exit()

split = int(0.8 * len(X))
X_train, X_val, y_train, y_val = X[:split], X[split:], y[:split], y[split:]

# 高效LSTM架构
model = Sequential([
    LSTM(CFG['lstm_units'][0], return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(CFG['drop_rate']),
    LSTM(CFG['lstm_units'][1]),
    Dropout(CFG['drop_rate']),
    Dense(CFG['future_steps'])
])

model.compile(optimizer=tf.keras.optimizers.Adam(CFG['lr']), loss='mae')  # 使用MAE损失
model.summary()

cb = [
    EarlyStopping(monitor='val_loss', patience=CFG['patience'], restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
]

history = model.fit(
    X_train, y_train, validation_data=(X_val, y_val),
    epochs=CFG['epochs'], batch_size=CFG['batch_size'], callbacks=cb, verbose=1
)

# === 7. 评估 & 可视化 (*** 此处为主要修改区域 ***) ===
all_preds, all_trues = [], []
for sid, g in df.groupby('station_ref_id'):
    if len(g) < CFG['past_steps'] + CFG['future_steps']:
        print(f"Skipping {sid}, not enough data for a test sample.")
        continue

    print(f'--- Evaluating {sid} ---')

    X_test_unscaled = g.tail(CFG['past_steps'] + CFG['future_steps'])
    
    # 获取特征名称列表
    feature_names = list(scaler_x_fitted.feature_names_in_)
    X_test_scaled = scaler_x_fitted.transform(X_test_unscaled[feature_names])
    X_test = X_test_scaled[:-CFG['future_steps']].reshape(1, CFG['past_steps'], -1)

    pred_scaled = model.predict(X_test, verbose=0)[0]
    pred = scaler_y_fitted.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    true = X_test_unscaled.tail(CFG['future_steps'])['load_discharge_delta'].values

    all_preds.append(pred)
    all_trues.append(true)

    # 计算指标
    # 避免真实值为0导致的MAPE无穷大问题
    true_safe = np.where(true == 0, 1e-6, true)
    mape = mean_absolute_percentage_error(true_safe, pred) * 100
    rmse = np.sqrt(mean_squared_error(true, pred))
    print(f'{sid}  MAPE={mape:.2f}%   RMSE={rmse:.2f}')
    
    # --- 新增的绘图逻辑 ---
    # 获取预测期的时间索引
    idx = X_test_unscaled['energy_date'].iloc[-CFG['future_steps']:].reset_index(drop=True)
    
    plt.figure(figsize=(15, 4))
    plt.plot(idx, true, label='Real', marker='.', linestyle='-')
    plt.plot(idx, pred, label='Pred', marker='.', linestyle='--')
    plt.title(f'Prediction for {sid}  |  MAPE={mape:.2f}%  RMSE={rmse:.2f}')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylabel('Load Discharge Delta')
    plt.xlabel('Date Time')
    plt.tight_layout()
    plt.savefig(f'output/prediction_optimized_{sid}.png') # 保存图像文件
    plt.close() # 关闭图形，防止在内存中累积

if all_trues:
    overall_mape = mean_absolute_percentage_error(np.concatenate(all_trues), np.concatenate(all_preds)) * 100
    overall_rmse = np.sqrt(mean_squared_error(np.concatenate(all_trues), np.concatenate(all_preds)))
    print(f'\n=== Overall Performance ===')
    print(f'Overall MAPE: {overall_mape:.2f}%')
    print(f'Overall RMSE: {overall_rmse:.2f}')

# === 8. 保存 ===
print('\nSaving model and scalers...')
model.save('output/model_optimized.h5')
joblib.dump(scaler_x_fitted, 'output/scaler_x_optimized.pkl')
joblib.dump(scaler_y_fitted, 'output/scaler_y_optimized.pkl')
joblib.dump(le, 'output/label_encoder_optimized.pkl')
print('All assets saved to output/')

# 可视化训练过程
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training History')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('output/training_history.png')
plt.close() # 关闭训练历史图

print("Process finished. Check the 'output' folder for models, scalers, and prediction plots.")