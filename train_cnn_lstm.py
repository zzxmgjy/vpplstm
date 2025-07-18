import os, gc, joblib, warnings
import pandas as pd
import numpy as np
import holidays
import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os
import sys
import warnings
import joblib

# --- PyTorch 相关导入 ---
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- 0. 超参数配置 (保持不变) ---
CFG = dict(
    past_steps    = 96 * 3,
    future_steps  = 96,
    lstm_hidden_dims = [128, 64], # PyTorch LSTM的隐藏层维度
    drop_rate     = 0.3,
    batch_size    = 256,
    epochs        = 150,
    patience      = 20,
    lr            = 1e-3,
    cb_iterations = 2500,
    cb_depth      = 10,
    cb_lr         = 0.03,
    cb_l2_reg     = 3,
    val_size      = 0.15,
    test_size     = 0.15
)

# === 1. 特征工程与内存优化 (保持不变) ===
def reduce_mem_usage(df, verbose=True):
    # (此函数与之前完全相同)
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max: df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max: df[col] = df[col].astype(np.int16)
                # ... 其他 int 类型 ...
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max: df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max: df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print(f'内存占用从 {start_mem:.2f} MB 减少到 {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

def add_advanced_features(df):
    # (此函数与之前完全相同)
    df = df.copy()
    df['hour'] = df['energy_date'].dt.hour; df['weekday'] = df['energy_date'].dt.weekday
    df['day'] = df['energy_date'].dt.day; df['month'] = df['energy_date'].dt.month
    df['weekofyear'] = df['energy_date'].dt.isocalendar().week.astype(int)
    df['temp_squared'] = df['temp'] ** 2
    df['is_holiday'] = df['energy_date'].isin(holidays.country_holidays('CN')).astype(int)
    df['load_discharge_delta'] = pd.to_numeric(df['load_discharge_delta'], errors='coerce')
    lags = [96, 96*7]; windows = [24, 96]
    for lag in lags: df[f'load_lag_{lag}'] = df.groupby('station_ref_id')['load_discharge_delta'].shift(lag)
    for win in windows:
        grouped_load = df.groupby('station_ref_id')['load_discharge_delta']
        df[f'load_rol_mean_{win}'] = grouped_load.rolling(window=win, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'load_rol_std_{win}'] = grouped_load.rolling(window=win, min_periods=1).std().reset_index(level=0, drop=True)
    df = df.fillna(method='ffill').dropna()
    return df

print("--- 1. 加载数据并进行高级特征工程 ---")
df = pd.read_csv('loaddata.csv', parse_dates=['energy_date'])
df = add_advanced_features(df)
df = reduce_mem_usage(df)
le = LabelEncoder()
df['station_enc'] = le.fit_transform(df['station_ref_id']).astype(np.int16)
gc.collect()

# === 2. 数据集划分 (保持不变) ===
print("\n--- 2. 按时间顺序划分数据集 ---")
train_df, test_df = train_test_split(df, test_size=CFG['test_size'] + CFG['val_size'], shuffle=False)
val_df, test_df = train_test_split(test_df, test_size=CFG['test_size'] / (CFG['test_size'] + CFG['val_size']), shuffle=False)
# (打印大小的代码省略)

# === 3. PyTorch LSTM 模型、数据集和训练循环 ===
print("\n--- 3. 准备 PyTorch LSTM 并开始训练 ---")

# 定义特征列
categorical_cols = ['is_holiday', 'hour', 'weekday', 'day', 'month', 'weekofyear', 'station_enc']
feature_cols = [col for col in df.columns if col not in ['energy_date', 'station_ref_id', 'load_discharge_delta']]
target_col = 'load_discharge_delta'

# 创建并拟合缩放器
scaler_x = StandardScaler(); scaler_y = StandardScaler()
scaler_x.fit(train_df[feature_cols]); scaler_y.fit(train_df[[target_col]])

# --- PyTorch 数据集类 ---
class TimeSeriesDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, scaler_x, scaler_y, past_steps, future_steps):
        self.past_steps, self.future_steps = past_steps, future_steps
        self.features = torch.tensor(scaler_x.transform(df[feature_cols]), dtype=torch.float32)
        self.targets = torch.tensor(scaler_y.transform(df[[target_col]]), dtype=torch.float32)

    def __len__(self):
        return len(self.features) - self.past_steps - self.future_steps + 1

    def __getitem__(self, index):
        past_end = index + self.past_steps
        future_end = past_end + self.future_steps
        return self.features[index:past_end], self.targets[past_end:future_end].squeeze(-1)

# --- PyTorch LSTM 模型类 ---
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, future_steps, drop_rate):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dims[0], batch_first=True)
        self.dropout1 = nn.Dropout(drop_rate)
        self.lstm2 = nn.LSTM(hidden_dims[0], hidden_dims[1], batch_first=True)
        self.dropout2 = nn.Dropout(drop_rate)
        self.fc = nn.Linear(hidden_dims[1], future_steps)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, (hn, _) = self.lstm2(x)
        x = self.dropout2(hn.squeeze(0)) # 使用最后一个隐藏状态进行预测
        return self.fc(x)

# --- 训练循环 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"将使用设备: {device}")

# 创建数据集和数据加载器
train_dataset = TimeSeriesDataset(train_df, feature_cols, target_col, scaler_x, scaler_y, CFG['past_steps'], CFG['future_steps'])
val_dataset = TimeSeriesDataset(val_df, feature_cols, target_col, scaler_x, scaler_y, CFG['past_steps'], CFG['future_steps'])
train_loader = DataLoader(train_dataset, batch_size=CFG['batch_size'], shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=CFG['batch_size'], shuffle=False, num_workers=2)

# 初始化模型、损失函数和优化器
model = LSTMModel(len(feature_cols), CFG['lstm_hidden_dims'], CFG['future_steps'], CFG['drop_rate']).to(device)
criterion = nn.L1Loss() # MAE Loss
optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'])
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)

# 训练
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(CFG['epochs']):
    model.train()
    for features, targets in train_loader:
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for features, targets in val_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            val_loss += criterion(outputs, targets).item()
    
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{CFG['epochs']}, Validation Loss: {avg_val_loss:.6f}")
    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'output_optimized_mem/pytorch_lstm_best.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= CFG['patience']:
            print("Early stopping triggered.")
            break
            
# 加载最佳模型权重
model.load_state_dict(torch.load('output_optimized_mem/pytorch_lstm_best.pth'))


# === 4. CatBoost 模型流程 (保持不变) ===
print("\n--- 4. 准备CatBoost数据并开始训练 ---")
# (此部分代码与之前完全相同)
X_train_cb, y_train_cb = train_df[feature_cols], train_df[target_col]
X_val_cb, y_val_cb = val_df[feature_cols], val_df[target_col]
categorical_features_indices = [feature_cols.index(col) for col in categorical_cols]
cb_model = cb.CatBoostRegressor(
    iterations=CFG['cb_iterations'], depth=CFG['cb_depth'], learning_rate=CFG['cb_lr'],
    l2_leaf_reg=CFG['cb_l2_reg'], loss_function='RMSE', verbose=100, random_seed=42
)
cb_model.fit(X_train_cb, y_train_cb, eval_set=(X_val_cb, y_val_cb), cat_features=categorical_features_indices, early_stopping_rounds=100, use_best_model=True)


# === 5. 计算模型权重 ===
print("\n--- 5. 在验证集上计算模型权重 ---")
# --- 获取 PyTorch LSTM 在验证集上的预测 ---
print("正在获取 PyTorch LSTM 在验证集上的预测...")
model.eval()
preds_lstm_val_list = []
true_lstm_val_list = []
with torch.no_grad():
    for features, targets in val_loader:
        features = features.to(device)
        outputs = model(features).cpu().numpy()
        preds_lstm_val_list.append(scaler_y.inverse_transform(outputs))
        true_lstm_val_list.append(scaler_y.inverse_transform(targets.cpu().numpy()))

preds_lstm_val = np.concatenate(preds_lstm_val_list)
true_lstm_val = np.concatenate(true_lstm_val_list)
var_lstm = np.var(true_lstm_val - preds_lstm_val)

# --- 获取 CatBoost 预测 ---
print("正在获取 CatBoost 在验证集上的预测...")
preds_cb_val = cb_model.predict(X_val_cb)
var_cb = np.var(y_val_cb.values - preds_cb_val)

# --- 计算权重 ---
inv_var_lstm = 1 / var_lstm if var_lstm > 0 else 0
inv_var_cb = 1 / var_cb if var_cb > 0 else 0
w_lstm = (inv_var_lstm / (inv_var_lstm + inv_var_cb)) if (inv_var_lstm + inv_var_cb) > 0 else 0.5
w_cb = 1.0 - w_lstm
print(f"\n验证集误差方差 (LSTM): {var_lstm:.4f}, (CatBoost): {var_cb:.4f}")
print(f"计算权重 -> LSTM: {w_lstm:.4f}, CatBoost: {w_cb:.4f}")


# === 6. 最终评估 ===
print("\n--- 6. 在测试集上评估组合模型 ---")
# --- 准备测试样本 ---
input_df = test_df.iloc[:CFG['past_steps'] + CFG['future_steps']]
test_dataset = TimeSeriesDataset(input_df, feature_cols, target_col, scaler_x, scaler_y, CFG['past_steps'], CFG['future_steps'])
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# --- 获取 PyTorch LSTM 预测 ---
features, _ = next(iter(test_loader))
features = features.to(device)
with torch.no_grad():
    pred_lstm_scaled = model(features).cpu().numpy()
pred_lstm = scaler_y.inverse_transform(pred_lstm_scaled).flatten()

# --- 获取 CatBoost 预测 ---
X_test_cb = input_df.iloc[CFG['past_steps']:][feature_cols]
true_values = input_df.iloc[CFG['past_steps']:][target_col].values
pred_cb = cb_model.predict(X_test_cb)

# --- 组合并评估 ---
pred_combined = (w_lstm * pred_lstm) + (w_cb * pred_cb)
true_values_safe = np.where(true_values == 0, 1e-6, true_values)
mape = mean_absolute_percentage_error(true_values_safe, pred_combined) * 100
rmse = np.sqrt(mean_squared_error(true_values, pred_combined))
mae = mean_absolute_error(true_values, pred_combined)
print(f"\n--- [PyTorch版] 单个测试样本上的最终性能 ---")
print(f"组合模型 MAPE: {mape:.2f}%, RMSE: {rmse:.2f}, MAE: {mae:.2f}")

# (可视化和保存模型的代码与之前类似，此处省略以保持简洁，但会保存 PyTorch 模型)
print("\n--- 7. 保存所有优化后的模型、缩放器和权重 ---")
# `torch.save` 用于保存PyTorch模型
torch.save(model.state_dict(), 'output_optimized_mem/pytorch_lstm_model.pth')
cb_model.save_model('output_optimized_mem/catboost_model.cbm')
joblib.dump(scaler_x, 'output_optimized_mem/scaler_x.pkl')
joblib.dump(scaler_y, 'output_optimized_mem/scaler_y.pkl')
joblib.dump(le, 'output_optimized_mem/label_encoder.pkl')
joblib.dump({'w_lstm': w_lstm, 'w_cb': w_cb}, 'output_optimized_mem/model_weights.pkl')
print("所有组件已成功保存到 'output_optimized_mem/' 文件夹中。")