import numpy as np
import pandas as pd
import holidays
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
# 导入 BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
import matplotlib.pyplot as plt
import os
import sys
import warnings
import joblib

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None
os.makedirs('output_cnn_lstm', exist_ok=True)

# ============================================================================
# 1 超参数
# ============================================================================
CFG = dict(
    freq='15T', jump_sigma=4, clip_sigma=3,
    split_ratio=(0.7, .15, .15),
    past_steps=96 * 5, future_steps=96,
    conv_filters=[64, 128], kernel_size=3, pool_size=2,
    lstm_units=[256, 128], drop_rate=.4,
    epochs=200, batch_size=256, patience=15,
    lr=1e-4, seed=42
)
np.random.seed(CFG['seed'])
tf.random.set_seed(CFG['seed'])
tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(2)

# ============================================================================
# 2 读取数据
# ============================================================================
SRC = 'loaddata.csv'
if not os.path.exists(SRC):
    sys.exit(f'❌ 未找到 {SRC}')
raw = pd.read_csv(SRC, parse_dates=['energy_date']).sort_values(
    ['station_ref_id', 'energy_date'])

# ============================================================================
# 3 清洗
# ============================================================================
def clean_one_station(df_sta, sid):
    log = {}
    df_sta = df_sta.sort_values('energy_date').drop_duplicates('energy_date')
    full_idx = pd.date_range(df_sta['energy_date'].min(),
                             df_sta['energy_date'].max(), freq=CFG['freq'])
    df_sta = df_sta.set_index('energy_date').reindex(full_idx)
    df_sta.index.name = 'energy_date'
    df_sta['station_ref_id'] = sid
    log['补行'] = int(df_sta['load_discharge_delta'].isna().sum())

    mask0 = df_sta['load_discharge_delta'] <= 0
    log['<=0→NaN'] = int(mask0.sum())
    df_sta.loc[mask0, 'load_discharge_delta'] = np.nan

    m, s = df_sta['load_discharge_delta'].mean(), df_sta['load_discharge_delta'].std()
    if s > 0:
        low, high = m - CFG['clip_sigma'] * s, m + CFG['clip_sigma'] * s
        clip_m = (df_sta['load_discharge_delta'] < low) | (df_sta['load_discharge_delta'] > high)
        log['3σ裁剪'] = int(clip_m.sum())
        df_sta['load_discharge_delta'] = df_sta['load_discharge_delta'].clip(low, high)

    diff = df_sta['load_discharge_delta'].diff().abs()
    jump_m = diff > CFG['jump_sigma'] * s
    log['跳变→NaN'] = int(jump_m.sum())
    df_sta.loc[jump_m, 'load_discharge_delta'] = np.nan

    miss_b = int(df_sta['load_discharge_delta'].isna().sum())
    df_sta['load_discharge_delta'] = df_sta['load_discharge_delta'].interpolate('time', limit_direction='both')
    miss_a = int(df_sta['load_discharge_delta'].isna().sum())
    log['缺失前/后'] = f'{miss_b}/{miss_a}'

    print('🧹', sid, ' | '.join([f'{k}:{v}' for k, v in log.items()]))
    return df_sta.reset_index()

df_clean = pd.concat([clean_one_station(g, sid) for sid, g in raw.groupby('station_ref_id')],
                     ignore_index=True)

# ============================================================================
# 4 时间切分
# ============================================================================
t0, t1 = df_clean['energy_date'].min(), df_clean['energy_date'].max()
span = (t1 - t0).total_seconds()
r_tr, r_v, _ = np.cumsum(CFG['split_ratio'])
t_tr_end, t_v_end = t0 + pd.Timedelta(seconds=span * r_tr), t0 + pd.Timedelta(seconds=span * r_v)

df_clean['set'] = 'test'
df_clean.loc[df_clean['energy_date'] <= t_tr_end, 'set'] = 'train'
df_clean.loc[(df_clean['energy_date'] > t_tr_end) & (df_clean['energy_date'] <= t_v_end), 'set'] = 'val'
print(df_clean['set'].value_counts())

# ============================================================================
# 5 特征工程
# ============================================================================
cn_holidays = holidays.country_holidays('CN')
def add_feat(df):
    df = df.copy()
    df['hour'] = df['energy_date'].dt.hour
    df['weekday'] = df['energy_date'].dt.weekday
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['sin_wday'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['cos_wday'] = np.cos(2 * np.pi * df['weekday'] / 7)
    df['temp_squared'] = df['temp'] ** 2
    df['is_holiday'] = df['energy_date'].isin(cn_holidays).astype(int)
    for lag in [1, 4, 24, 96]:
        df[f'load_lag{lag}'] = df.groupby('station_ref_id')['load_discharge_delta'].shift(lag)
    for win in [4, 24, 96]:
        grp = df.groupby('station_ref_id')['load_discharge_delta']
        df[f'load_ma{win}'] = grp.rolling(win, 1).mean().reset_index(level=0, drop=True)
        df[f'load_std{win}'] = grp.rolling(win, 1).std().reset_index(level=0, drop=True)
    return df
df_feat = add_feat(df_clean)

# ============================================================================
# 6 编码
# ============================================================================
df_feat['station_enc'] = LabelEncoder().fit_transform(df_feat['station_ref_id'])

# ============================================================================
# 7 SafeStandardScaler + make_dataset
# ============================================================================
feature_cols = ['temp', 'temp_squared', 'humidity', 'windSpeed',
                'load_lag1', 'load_lag4', 'load_lag24', 'load_lag96',
                'load_ma4', 'load_ma24', 'load_ma96',
                'load_std4', 'load_std24', 'load_std96',
                'is_holiday', 'sin_hour', 'cos_hour', 'sin_wday', 'cos_wday',
                'station_enc']
target_col = 'load_discharge_delta'

class SafeStandardScaler(StandardScaler):
    def fit(self, X, y=None):
        super().fit(X, y)
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        return self

def make_dataset(data_set, past, fut, df_src):
    sub = df_src[df_src['set'].isin(data_set)].copy().sort_values(['station_enc', 'energy_date'])
    sub[feature_cols + [target_col]] = sub[feature_cols + [target_col]].astype(float)
    sub[feature_cols + [target_col]] = sub.groupby('station_enc')[feature_cols + [target_col]]\
        .apply(lambda g: g.set_index(sub.loc[g.index, 'energy_date'])
                          .interpolate('time', limit_direction='both'))\
        .reset_index(drop=True)
    sub[feature_cols + [target_col]] = sub[feature_cols + [target_col]].fillna(0)

    scaler_x, scaler_y = SafeStandardScaler(), SafeStandardScaler()
    X_scaled = scaler_x.fit_transform(sub[feature_cols])
    y_scaled = scaler_y.fit_transform(sub[[target_col]])

    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    y_scaled = np.nan_to_num(y_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    sub_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=sub.index)
    sub_scaled[target_col] = y_scaled
    sub_scaled['station_enc'] = sub['station_enc'].values
    sub_scaled['energy_date'] = sub['energy_date'].values

    Xs, ys = [], []
    for sid in sub_scaled['station_enc'].unique():
        d = sub_scaled[sub_scaled['station_enc'] == sid]
        if len(d) < past + fut:
            continue
        xf, yf = d[feature_cols].values, d[[target_col]].values
        for i in range(len(d) - past - fut + 1):
            Xs.append(xf[i:i + past])
            ys.append(yf[i + past:i + past + fut, 0])
    Xs, ys = np.array(Xs), np.array(ys)
    ok = np.isfinite(Xs).all((1, 2)) & np.isfinite(ys).all(1)
    if ok.sum() < len(ok):
        print(f'[NanFilter] {data_set} remove {len(ok) - ok.sum()} bad samples')
    Xs, ys = Xs[ok], ys[ok]
    if len(Xs) == 0:
        raise ValueError(f'{data_set} 无有效样本')
    return Xs, ys, scaler_x, scaler_y

X_train, y_train, scaler_x, scaler_y = make_dataset({'train'}, CFG['past_steps'], CFG['future_steps'], df_feat)
X_val, y_val, _, _ = make_dataset({'val'}, CFG['past_steps'], CFG['future_steps'], df_feat)
X_test, y_test, _, _ = make_dataset({'test'}, CFG['past_steps'], CFG['future_steps'], df_feat)

print(f'Train:{X_train.shape}  Val:{X_val.shape}  Test:{X_test.shape}')

# 二次断言
def chk(name, arr):
    assert np.isfinite(arr).all(), f'{name} 存在 NaN/Inf'
chk('X_train', X_train); chk('y_train', y_train)

# ============================================================================
# 8 模型
# ============================================================================
# 【【【代码修改处】】】
model = Sequential([
    Conv1D(CFG['conv_filters'][0], CFG['kernel_size'], activation='relu', padding='same',
           input_shape=(CFG['past_steps'], X_train.shape[2])),
    BatchNormalization(),  # 修改：在卷积层后添加批量归一化
    MaxPooling1D(CFG['pool_size']),
    Dropout(CFG['drop_rate']),

    Conv1D(CFG['conv_filters'][1], CFG['kernel_size'], activation='relu', padding='same'),
    BatchNormalization(),  # 修改：在卷积层后添加批量归一化
    MaxPooling1D(CFG['pool_size']),
    Dropout(CFG['drop_rate']),

    LSTM(CFG['lstm_units'][0], return_sequences=True),
    Dropout(CFG['drop_rate']),
    LSTM(CFG['lstm_units'][1]),
    Dropout(CFG['drop_rate']),
    Dense(CFG['future_steps'])
])
# 【【【代码修改结束】】】

model.compile(optimizer=tf.keras.optimizers.Adam(CFG['lr'], clipnorm=1.0),
              loss=tf.keras.losses.Huber(delta=1.0))
model.summary()

history = model.fit(X_train, y_train, epochs=CFG['epochs'], batch_size=CFG['batch_size'],
                    validation_data=(X_val, y_val),
                    callbacks=[TerminateOnNaN(),
                               EarlyStopping('val_loss', patience=CFG['patience'], restore_best_weights=True),
                               ReduceLROnPlateau('val_loss', factor=.5, patience=5, min_lr=1e-5)],
                    verbose=1)

# ============================================================================
# 9 预测/评估（预测前再断言一次）
# ============================================================================
chk('X_test(pre)', X_test)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

pred_scaled = model.predict(X_test, verbose=0)
pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
finite = np.isfinite(pred) & np.isfinite(true)
pred, true = pred[finite], true[finite]

mape = mean_absolute_percentage_error(np.where(true == 0, 1e-6, true), pred) * 100
rmse = np.sqrt(mean_squared_error(true, pred))
print(f'\n=== Test Set ===  MAPE:{mape:.2f}%  RMSE:{rmse:.2f}')

# ============================================================================
# 10 保存
# ============================================================================
model.save('output_cnn_lstm/model_cnn_lstm.h5')
joblib.dump(scaler_x, 'output_cnn_lstm/scaler_x.pkl')
joblib.dump(scaler_y, 'output_cnn_lstm/scaler_y.pkl')
print('✅  模型与Scaler已保存')

plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.legend(); plt.title('Training History'); plt.tight_layout()
plt.savefig('output_cnn_lstm/training_history.png'); plt.close()
print('🎉  全流程结束，详见 output_cnn_lstm/ 目录')
