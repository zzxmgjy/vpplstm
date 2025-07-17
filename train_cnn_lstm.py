# ============================================================================
# 0. 依赖 & 环境
# ============================================================================
import os, joblib, warnings, sys
import pandas as pd
import numpy as np
import holidays
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D,
                                     LSTM, Dense, Dropout)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.makedirs('output_cnn_lstm', exist_ok=True)

# ============================================================================
# 1. 超参数配置
# ============================================================================
CFG = dict(
    # --- 序列步长 ---
    past_steps   = 96 * 5,    # 过去 5 天 (15min 采样 -> 96 点/天)
    future_steps = 96,        # 未来 1 天
    # --- CNN ---
    conv_filters = [32, 64],
    kernel_size  = 3,
    pool_size    = 2,
    # --- LSTM ---
    lstm_units   = [256, 128],
    drop_rate    = 0.4,
    # --- 训练 ---
    batch_size   = 256,
    epochs       = 150,
    patience     = 15,
    lr           = 8e-4,
    # --- 其他 ---
    seed         = 42
)

np.random.seed(CFG['seed'])
tf.random.set_seed(CFG['seed'])

# ============================================================================
# 2. 读数
# ============================================================================
csv_file = 'loaddata.csv'
if not os.path.exists(csv_file):
    sys.exit(f'未找到 {csv_file}，请将数据文件放在脚本同目录下！')

df = pd.read_csv(csv_file, parse_dates=['energy_date'])
df = df.sort_values(['station_ref_id', 'energy_date'])

# ============================================================================
# 3. 特征工程
# ============================================================================
cn_holidays = holidays.country_holidays('CN')

def add_time_feat(_df: pd.DataFrame) -> pd.DataFrame:
    df_ = _df.copy()
    # 基础时间特征
    df_['hour']    = df_['energy_date'].dt.hour
    df_['weekday'] = df_['energy_date'].dt.weekday
    df_['day']     = df_['energy_date'].dt.day
    df_['month']   = df_['energy_date'].dt.month

    # 处理目标列
    df_['load_discharge_delta'] = pd.to_numeric(df_['load_discharge_delta'],
                                                errors='coerce')
    df_['load_discharge_delta'] = df_.groupby('station_ref_id')[
        'load_discharge_delta'].ffill()

    # 滞后特征
    lags = [1, 4, 24, 96]
    for lag in lags:
        df_[f'load_lag{lag}'] = df_.groupby('station_ref_id')[
            'load_discharge_delta'].shift(lag)

    # 移动统计
    windows = [4, 24, 96]
    for win in windows:
        grp = df_.groupby('station_ref_id')['load_discharge_delta']
        df_[f'load_ma{win}']  = grp.rolling(win, 1).mean().reset_index(
            level=0, drop=True)
        df_[f'load_std{win}'] = grp.rolling(win, 1).std().reset_index(
            level=0, drop=True)

    # 节假日
    df_['is_holiday'] = df_['energy_date'].isin(cn_holidays).astype(int)

    # 周期 sin / cos
    df_['sin_hour'] = np.sin(2*np.pi*df_['hour']/24)
    df_['cos_hour'] = np.cos(2*np.pi*df_['hour']/24)
    df_['sin_wday'] = np.sin(2*np.pi*df_['weekday']/7)
    df_['cos_wday'] = np.cos(2*np.pi*df_['weekday']/7)

    # 温度二次项
    df_['temp_squared'] = df_['temp'] ** 2

    return df_

df = add_time_feat(df)

# ============================================================================
# 4. 清洗 & NaN 处理
# ============================================================================
df = df.set_index('energy_date')

# 仅对目标列做 3σ 裁剪
for sid in df['station_ref_id'].unique():
    mask = df['station_ref_id'] == sid
    col_data = df.loc[mask, 'load_discharge_delta']
    mean, std = col_data.mean(), col_data.std()
    if std > 0:
        low, high = mean - 3*std, mean + 3*std
        df.loc[mask, 'load_discharge_delta'] = col_data.clip(low, high)

df = df.fillna(method='ffill').dropna()
df = df.reset_index()

# ============================================================================
# 5. LabelEncoder
# ============================================================================
le = LabelEncoder()
df['station_enc'] = le.fit_transform(df['station_ref_id'])

# ============================================================================
# 6. 滑窗样本制作
# ============================================================================
feature_cols = [
    'temp', 'temp_squared', 'humidity', 'windSpeed',
    'load_lag1', 'load_lag4', 'load_lag24', 'load_lag96',
    'load_ma4', 'load_ma24', 'load_ma96',
    'load_std4', 'load_std24', 'load_std96',
    'is_holiday', 'sin_hour', 'cos_hour', 'sin_wday', 'cos_wday',
    'station_enc'
]
target_col = 'load_discharge_delta'

def make_dataset(data: pd.DataFrame,
                 past_steps: int,
                 future_steps: int):
    data = data.sort_values(['station_enc', 'energy_date'])\
               .set_index('energy_date')

    # 标准化器
    scaler_x, scaler_y = StandardScaler(), StandardScaler()

    # 保证全为数值
    for col in feature_cols + [target_col]:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.fillna(method='ffill').dropna()
    if data.empty:
        return np.array([]), np.array([]), None, None

    X_scaled = scaler_x.fit_transform(data[feature_cols])
    y_scaled = scaler_y.fit_transform(data[[target_col]])

    scaled_df = pd.DataFrame(X_scaled,
                             columns=feature_cols,
                             index=data.index)
    scaled_df[target_col] = y_scaled
    scaled_df['station_enc'] = data['station_enc'].values  # 保留编码列

    Xs, ys = [], []
    for sid in scaled_df['station_enc'].unique():
        sub_df = scaled_df[scaled_df['station_enc'] == sid]
        if len(sub_df) < past_steps + future_steps:
            continue
        X_arr = sub_df[feature_cols].values
        y_arr = sub_df[[target_col]].values
        for i in range(len(sub_df) - past_steps - future_steps + 1):
            Xs.append(X_arr[i:i+past_steps])
            ys.append(y_arr[i+past_steps:i+past_steps+future_steps, 0])

    return np.array(Xs), np.array(ys), scaler_x, scaler_y

print('=== 构造滑窗样本 ===')
X, y, scaler_x, scaler_y = make_dataset(df,
                                        CFG['past_steps'],
                                        CFG['future_steps'])
if len(X) == 0:
    sys.exit("滑窗样本为空，请检查数据量或参数！")

# 划分训练 / 验证
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

print(f'Train shape: {X_train.shape},  Val shape: {X_val.shape}')

# ============================================================================
# 7. CNN-LSTM 模型
# ============================================================================
model = Sequential([
    # CNN ①
    Conv1D(filters=CFG['conv_filters'][0],
           kernel_size=CFG['kernel_size'],
           activation='relu',
           padding='same',
           input_shape=(X.shape[1], X.shape[2])),
    MaxPooling1D(pool_size=CFG['pool_size']),
    Dropout(CFG['drop_rate']),
    # CNN ②
    Conv1D(filters=CFG['conv_filters'][1],
           kernel_size=CFG['kernel_size'],
           activation='relu',
           padding='same'),
    MaxPooling1D(pool_size=CFG['pool_size']),
    Dropout(CFG['drop_rate']),
    # LSTM 堆叠
    LSTM(CFG['lstm_units'][0], return_sequences=True),
    Dropout(CFG['drop_rate']),
    LSTM(CFG['lstm_units'][1]),
    Dropout(CFG['drop_rate']),
    # Dense 输出未来 96 步
    Dense(CFG['future_steps'])
])

model.compile(optimizer=tf.keras.optimizers.Adam(CFG['lr']),
              loss='mae')
model.summary()

# ============================================================================
# 8. 训练
# ============================================================================
callbacks = [
    EarlyStopping(monitor='val_loss',
                  patience=CFG['patience'],
                  restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss',
                      factor=0.5,
                      patience=5,
                      min_lr=1e-5)
]
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=CFG['epochs'],
    batch_size=CFG['batch_size'],
    callbacks=callbacks,
    verbose=1
)

# ============================================================================
# 9. 逐站点评估 & 可视化
# ============================================================================
all_preds, all_trues = [], []
for sid, g in df.groupby('station_ref_id'):
    if len(g) < CFG['past_steps'] + CFG['future_steps']:
        print(f'[Skip] {sid} 数据不足')
        continue

    test_block = g.tail(CFG['past_steps'] + CFG['future_steps'])
    X_test_scaled = scaler_x.transform(test_block[feature_cols])
    X_test = X_test_scaled[:-CFG['future_steps']]\
             .reshape(1, CFG['past_steps'], -1)

    pred_scaled = model.predict(X_test, verbose=0)[0]
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1,1)).flatten()
    true = test_block.tail(CFG['future_steps'])[target_col].values

    # 保存整体数组
    all_preds.append(pred)
    all_trues.append(true)

    # 指标
    true_safe = np.where(true == 0, 1e-6, true)
    mape = mean_absolute_percentage_error(true_safe, pred) * 100
    rmse = np.sqrt(mean_squared_error(true, pred))
    print(f'{sid}  |  MAPE={mape:.2f}%  RMSE={rmse:.2f}')

    # 绘图
    idx = test_block['energy_date'].iloc[-CFG['future_steps']:].reset_index(drop=True)
    plt.figure(figsize=(15,4))
    plt.plot(idx, true, label='Real', marker='.')
    plt.plot(idx, pred, label='Pred', marker='.')
    plt.title(f'{sid}  |  MAPE={mape:.2f}%  RMSE={rmse:.2f}')
    plt.legend(); plt.grid(True, linestyle='--', linewidth=.5)
    plt.ylabel('Load Discharge Delta'); plt.xlabel('DateTime')
    plt.tight_layout()
    plt.savefig(f'output/pred_{sid}.png')
    plt.close()

# 总体指标
if all_trues:
    overall_mape = mean_absolute_percentage_error(
        np.concatenate(all_trues),
        np.concatenate(all_preds)) * 100
    overall_rmse = np.sqrt(mean_squared_error(
        np.concatenate(all_trues),
        np.concatenate(all_preds)))
    print('\n=== Overall ===')
    print(f'MAPE : {overall_mape:.2f}%')
    print(f'RMSE : {overall_rmse:.2f}')

# ============================================================================
# 10. 资产保存
# ============================================================================
print('\nSaving model & scalers...')
model.save('output/model_cnn_lstm_cnn.h5')
joblib.dump(scaler_x, 'output/scaler_x_cnn.pkl')
joblib.dump(scaler_y, 'output/scaler_y_cnn.pkl')
joblib.dump(le,        'output/label_encoder_cnn.pkl')

# 训练曲线
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Training History')
plt.ylabel('MAE'); plt.xlabel('Epoch')
plt.legend(); plt.tight_layout()
plt.savefig('output/training_history_cnn.png')
plt.close()

print('全部完成！查看 output 文件夹获取模型与图表。')