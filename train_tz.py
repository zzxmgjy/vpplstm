import os, joblib, warnings
import pandas as pd
import numpy as np
import holidays
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import matplotlib.pyplot as plt

# --- PyTorch 相关导入 ---
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

warnings.filterwarnings("ignore")
os.makedirs('output_pytorch', exist_ok=True) # 使用新的输出文件夹

# === 0. 优化超参 (与TF版本保持一致) ===
CFG = dict(
    past_steps    = 96 * 5,
    future_steps  = 96,
    hidden_dims   = [128, 128], # PyTorch LSTM 的隐藏层维度
    drop_rate     = 0.2401,
    batch_size    = 256,
    epochs        = 150,
    patience      = 15,
    lr            = 2.16485e-4
)

# === 1. 读数 (保持不变) ===
df = pd.read_csv('loaddata.csv', parse_dates=['energy_date'])
df = df.sort_values(['station_ref_id', 'energy_date'])

# === 2. 核心特征工程 (保持不变) ===
cn_holidays = holidays.country_holidays('CN')

def add_time_feat(df):
    # (此函数与TF版本完全相同)
    df = df.copy()
    df['hour'] = df['energy_date'].dt.hour
    df['weekday'] = df['energy_date'].dt.weekday
    df['day'] = df['energy_date'].dt.day
    df['month'] = df['energy_date'].dt.month
    df['load_discharge_delta'] = pd.to_numeric(df['load_discharge_delta'], errors='coerce')
    df['load_discharge_delta'] = df.groupby('station_ref_id')['load_discharge_delta'].ffill()
    lags = [1, 4, 24, 96]
    for lag in lags: df[f'load_lag{lag}'] = df.groupby('station_ref_id')['load_discharge_delta'].shift(lag)
    windows = [4, 24, 96]
    for win in windows:
        grouped_load = df.groupby('station_ref_id')['load_discharge_delta']
        df[f'load_ma{win}'] = grouped_load.rolling(win, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'load_std{win}'] = grouped_load.rolling(win, min_periods=1).std().reset_index(level=0, drop=True)
    df['is_holiday'] = df['energy_date'].isin(cn_holidays).astype(int)
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['sin_wday'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['cos_wday'] = np.cos(2 * np.pi * df['weekday'] / 7)
    df['temp_squared'] = df['temp'] ** 2
    return df

df = add_time_feat(df)

# === 3. 数据清洗 & NaN 处理 (保持不变) ===
df = df.set_index('energy_date')
for sid in df['station_ref_id'].unique():
    station_mask = df['station_ref_id'] == sid
    col_data = df.loc[station_mask, 'load_discharge_delta']
    mean, std = col_data.mean(), col_data.std()
    if std > 0:
        low, high = mean - 3 * std, mean + 3 * std
        df.loc[station_mask, 'load_discharge_delta'] = col_data.clip(lower=low, upper=high)
df = df.fillna(method='ffill').dropna()
df = df.reset_index()

# === 4. station 编码 (保持不变) ===
le = LabelEncoder()
df['station_enc'] = le.fit_transform(df['station_ref_id'])

# === 5. 构造数据集 (PyTorch 版本) ===
def make_dataset_pytorch(data, past_steps, future_steps):
    # (此函数逻辑与TF版本完全相同，只是为了清晰区分)
    data = data.sort_values(['station_enc', 'energy_date']).set_index('energy_date')
    feature_cols = [
        'temp', 'temp_squared', 'humidity', 'windSpeed', 'load_lag1', 'load_lag4', 'load_lag24', 'load_lag96',
        'load_ma4', 'load_ma24', 'load_ma96', 'load_std4', 'load_std24', 'load_std96', 'is_holiday',
        'sin_hour', 'cos_hour', 'sin_wday', 'cos_wday', 'station_enc'
    ]
    target_col = 'load_discharge_delta'
    scaler_x, scaler_y = StandardScaler(), StandardScaler()
    for col in feature_cols + [target_col]: data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.fillna(method='ffill').dropna()
    if data.empty: return None, None, None, None
    
    X_scaled = scaler_x.fit_transform(data[feature_cols])
    y_scaled = scaler_y.fit_transform(data[[target_col]])
    
    scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=data.index)
    scaled_df[target_col] = y_scaled
    
    Xs, ys = [], []
    for sid_enc in scaled_df['station_enc'].unique():
        station_df = scaled_df[scaled_df['station_enc'] == sid_enc]
        if len(station_df) < past_steps + future_steps: continue
        X_station, y_station = station_df[feature_cols].values, station_df[[target_col]].values
        for i in range(len(X_station) - past_steps - future_steps + 1):
            Xs.append(X_station[i:i + past_steps])
            ys.append(y_station[i + past_steps:i + past_steps + future_steps, 0])
            
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32), scaler_x, scaler_y, feature_cols

# === 6. 训练优化模型 (PyTorch 版本) ===
print('=== Training Optimized Model (PyTorch Version) ===')
X, y, scaler_x_fitted, scaler_y_fitted, feature_cols = make_dataset_pytorch(df, CFG['past_steps'], CFG['future_steps'])

if X is None:
    print("制作滑窗样本后数据为空，请检查数据量。")
    exit()

# 划分数据集
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# --- PyTorch 数据集和数据加载器 ---
train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
train_loader = DataLoader(train_dataset, batch_size=CFG['batch_size'], shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=CFG['batch_size'], shuffle=False, num_workers=2)

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
        # PyTorch LSTM 返回 (output, (h_n, c_n))
        # 我们取最后一个时间步的输出 hn 进行预测
        _, (hn, _) = self.lstm2(x) 
        # hn 的形状是 (num_layers, batch, hidden_size)，我们需要调整一下
        x = self.dropout2(hn.squeeze(0))
        return self.fc(x)
        
# --- PyTorch 训练循环 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"将使用设备: {device}")

model = LSTMModel(len(feature_cols), CFG['hidden_dims'], CFG['future_steps'], CFG['drop_rate']).to(device)
criterion = nn.L1Loss() # MAE Loss
optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'])
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

best_val_loss = float('inf')
patience_counter = 0
history = {'loss': [], 'val_loss': []}

for epoch in range(CFG['epochs']):
    model.train()
    epoch_loss = 0
    for features, targets in train_loader:
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_train_loss = epoch_loss / len(train_loader)
    history['loss'].append(avg_train_loss)
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for features, targets in val_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            val_loss += criterion(outputs, targets).item()
    
    avg_val_loss = val_loss / len(val_loader)
    history['val_loss'].append(avg_val_loss)
    print(f"Epoch {epoch+1}/{CFG['epochs']}, Train Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")
    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'output_pytorch/pytorch_model_best.pth')
        patience_counter = 0
        print(f"Validation loss decreased. Saving model.")
    else:
        patience_counter += 1
        if patience_counter >= CFG['patience']:
            print("Early stopping triggered.")
            break

# 加载最佳模型权重
model.load_state_dict(torch.load('output_pytorch/pytorch_model_best.pth'))


# === 7. 评估 & 可视化 (PyTorch 版本) ===
all_preds, all_trues = [], []
model.eval()
with torch.no_grad():
    for sid, g in df.groupby('station_ref_id'):
        if len(g) < CFG['past_steps'] + CFG['future_steps']:
            print(f"Skipping {sid}, not enough data for a test sample.")
            continue

        print(f'--- Evaluating {sid} ---')
        X_test_unscaled = g.tail(CFG['past_steps'] + CFG['future_steps'])
        
        # 使用之前 fit 好的 scaler
        X_test_scaled = scaler_x_fitted.transform(X_test_unscaled[feature_cols])
        X_test = X_test_scaled[:-CFG['future_steps']]
        
        # 转换为 PyTorch Tensor 并移动到设备
        X_test_tensor = torch.from_numpy(X_test.astype(np.float32)).unsqueeze(0).to(device)
        
        pred_scaled = model(X_test_tensor).cpu().numpy().flatten()
        pred = scaler_y_fitted.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        true = X_test_unscaled.tail(CFG['future_steps'])['load_discharge_delta'].values

        all_preds.append(pred)
        all_trues.append(true)

        true_safe = np.where(true == 0, 1e-6, true)
        mape = mean_absolute_percentage_error(true_safe, pred) * 100
        rmse = np.sqrt(mean_squared_error(true, pred))
        print(f'{sid}  MAPE={mape:.2f}%   RMSE={rmse:.2f}')
        
        idx = X_test_unscaled['energy_date'].iloc[-CFG['future_steps']:].reset_index(drop=True)
        plt.figure(figsize=(15, 4))
        plt.plot(idx, true, label='Real', marker='.', linestyle='-')
        plt.plot(idx, pred, label='Pred', marker='.', linestyle='--')
        plt.title(f'Prediction for {sid} | MAPE={mape:.2f}% RMSE={rmse:.2f}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'output_pytorch/prediction_optimized_{sid}.png')
        plt.close()

if all_trues:
    overall_mape = mean_absolute_percentage_error(np.concatenate(all_trues), np.concatenate(all_preds)) * 100
    overall_rmse = np.sqrt(mean_squared_error(np.concatenate(all_trues), np.concatenate(all_preds)))
    print(f'\n=== Overall Performance ===')
    print(f'Overall MAPE: {overall_mape:.2f}%')
    print(f'Overall RMSE: {overall_rmse:.2f}')

# === 8. 保存 (PyTorch 版本) ===
print('\nSaving model and scalers...')
torch.save(model.state_dict(), 'output_pytorch/model_optimized.pth')
joblib.dump(scaler_x_fitted, 'output_pytorch/scaler_x_optimized.pkl')
joblib.dump(scaler_y_fitted, 'output_pytorch/scaler_y_optimized.pkl')
joblib.dump(le, 'output_pytorch/label_encoder_optimized.pkl')
print('All assets saved to output_pytorch/')

# 可视化训练过程
plt.figure(figsize=(10, 5))
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training History')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('output_pytorch/training_history.png')
plt.close()

print("Process finished. Check the 'output_pytorch' folder for models, scalers, and prediction plots.")