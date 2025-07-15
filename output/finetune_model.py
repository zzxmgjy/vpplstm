# finetune_model.py (v2 - with Historical Context)
import os
import joblib
import warnings
import pandas as pd
import numpy as np
import holidays
import tensorflow as tf

warnings.filterwarnings("ignore")
print(f"TensorFlow Version: {tf.__version__}")

# === 0. 配置与路径 ===
OUTPUT_DIR = 'output'
MODEL_PATH = os.path.join(OUTPUT_DIR, 'model_global.h5')
SCALER_X_PATH = os.path.join(OUTPUT_DIR, 'scaler_x_global.pkl')
SCALER_Y_PATH = os.path.join(OUTPUT_DIR, 'scaler_y_global.pkl')
LABEL_ENCODER_PATH = os.path.join(OUTPUT_DIR, 'label_encoder.pkl')

# --- 新增：历史上下文文件的路径 ---
# **重要**: 这个文件应包含您原始训练数据中最后的 480 行。
HISTORICAL_CONTEXT_PATH = 'load.csv' 
INCREMENTAL_DATA_PATH = 'out/incremental_data.csv'

# --- 超参数 ---
FINETUNE_LEARNING_RATE = 1e-5 
FINETUNE_EPOCHS = 20
FINETUNE_BATCH_SIZE = 256
PAST_STEPS = 480
FUTURE_STEPS = 96

# === 1. 复刻与原始训练完全一致的函数 ===
# (add_time_feat 和 make_incremental_dataset 函数保持不变, 此处为简洁省略，实际代码中需要保留)
cn_holidays = holidays.country_holidays('CN')
def add_time_feat(df):
    df = df.copy()
    df['energy_date'] = pd.to_datetime(df['energy_date'])
    df['hour'] = df['energy_date'].dt.hour
    df['weekday'] = df['energy_date'].dt.weekday
    df['day'] = df['energy_date'].dt.day
    df['month'] = df['energy_date'].dt.month
    for lag in [1, 4, 96, 192]:
        df[f'load_lag{lag}'] = df.groupby('station_ref_id')['load_discharge_delta'].shift(lag)
    for win in [4, 12, 96, 192]:
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

def make_incremental_dataset(data, scaler_x, scaler_y, past_steps, future_steps):
    data = data.sort_values(['station_enc', 'energy_date']).set_index('energy_date')
    feature_cols = scaler_x.feature_names_in_
    target_col = 'load_discharge_delta'
    X_scaled = scaler_x.transform(data[feature_cols])
    y_scaled = scaler_y.transform(data[[target_col]])
    scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=data.index)
    scaled_df[target_col] = y_scaled
    scaled_df['station_enc'] = data['station_enc'].values
    Xs, ys = [], []
    for sid_enc in scaled_df['station_enc'].unique():
        station_df = scaled_df[scaled_df['station_enc'] == sid_enc]
        X_station = station_df[feature_cols].values
        y_station = station_df[[target_col]].values
        if len(X_station) >= past_steps + future_steps:
            for i in range(len(X_station) - past_steps - future_steps + 1):
                Xs.append(X_station[i:i + past_steps])
                ys.append(y_station[i + past_steps:i + past_steps + future_steps, 0])
    return np.array(Xs), np.array(ys)


# === 2. 主执行流程 ===
def run_finetuning():
    print("--- 启动模型增量训练流程 (V2 - 带历史上下文) ---")

    # --- 步骤 1: 加载现有资产 ---
    print("步骤 1: 加载现有模型和预处理器...")
    # ... (代码不变) ...
    try:
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'mae': tf.keras.losses.MeanAbsoluteError})
        scaler_x = joblib.load(SCALER_X_PATH)
        scaler_y = joblib.load(SCALER_Y_PATH)
        le = joblib.load(LABEL_ENCODER_PATH)
        print("模型和预处理器加载成功。")
    except Exception as e:
        print(f"加载资产时发生致命错误: {e}")
        return

    # --- 步骤 2: 加载上下文和增量数据 ---
    print(f"\n步骤 2: 加载数据...")
    if not os.path.exists(HISTORICAL_CONTEXT_PATH):
        print(f"错误：找不到历史上下文文件 '{HISTORICAL_CONTEXT_PATH}'。")
        return
    if not os.path.exists(INCREMENTAL_DATA_PATH):
        print(f"错误：找不到增量数据文件 '{INCREMENTAL_DATA_PATH}'。")
        return
        
    df_context = pd.read_csv(HISTORICAL_CONTEXT_PATH, parse_dates=['energy_date'])
    df_incremental = pd.read_csv(INCREMENTAL_DATA_PATH, parse_dates=['energy_date'])
    
    print(f"加载了 {len(df_context)} 条历史上下文数据。")
    print(f"加载了 {len(df_incremental)} 条新的增量数据。")
    
    # --- 步骤 3: 合并数据并进行特征工程 ---
    print("\n步骤 3: 合并数据以进行统一的特征工程...")
    df_combined = pd.concat([df_context, df_incremental], ignore_index=True).sort_values(['station_ref_id', 'energy_date']).reset_index(drop=True)
    
    print("对合并后的数据应用特征工程...")
    df_featured = add_time_feat(df_combined)
    
    # 清理NaN值
    df_featured = df_featured.fillna(method='ffill')
    
    # **关键一步**：在创建滑窗样本前，只保留我们感兴趣的增量数据部分
    # 我们只取原始增量数据部分进行处理，因为上下文数据已经训练过了
    # 通过索引来安全地切片
    start_index_of_incremental = len(df_context)
    df_new_processed = df_featured.iloc[start_index_of_incremental:].copy()
    
    # 检查清理后是否还有数据
    if df_new_processed.isnull().values.any():
        print("警告：即使有历史上下文，处理后的增量数据中仍存在NaN值。正在丢弃这些行...")
        df_new_processed.dropna(inplace=True)
        print(f"丢弃后剩余 {len(df_new_processed)} 行。")

    if df_new_processed.empty:
        print("错误：处理后，没有剩余的增量数据可用于训练。")
        return
        
    # --- 步骤 4: 编码并创建数据集 ---
    print("\n步骤 4: 对新数据部分进行编码和滑窗...")
    new_stations = set(df_new_processed['station_ref_id'].unique()) - set(le.classes_)
    if new_stations:
        print(f"错误：增量数据中包含模型未见过的站点ID: {new_stations}。")
        return
    df_new_processed['station_enc'] = le.transform(df_new_processed['station_ref_id'])

    X_new, y_new = make_incremental_dataset(df_new_processed, scaler_x, scaler_y, PAST_STEPS, FUTURE_STEPS)
    
    if len(X_new) == 0:
        print("警告：增量数据量不足，无法构成任何一个完整的训练样本。")
        print(f"需要至少 {PAST_STEPS + FUTURE_STEPS} 条连续的增量数据才能生成一个样本。")
        return
    
    print(f"成功为增量数据创建了 {len(X_new)} 个训练样本。")

    # --- 步骤 5 & 6 (编译、微调、保存) 与之前一致 ---
    print("\n步骤 5: 使用低学习率编译模型并开始微调...")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=FINETUNE_LEARNING_RATE), loss='mae')
    model.fit(X_new, y_new, epochs=FINETUNE_EPOCHS, batch_size=FINETUNE_BATCH_SIZE, verbose=1)
    
    print(f"\n步骤 6: 保存更新后的模型到 '{MODEL_PATH}'...")
    model.save(MODEL_PATH)
    print("\n--- 增量训练流程成功结束！ ---")


if __name__ == '__main__':
    run_finetuning()