# 双输出预测更新说明 (Energy + Power Prediction)

## 更新概述
已成功将系统从单一电量预测更新为双输出预测，同时预测：
1. **forward_total_active_energy** (正向有功电能)
2. **total_active_power** (总有功功率)

## 主要变更

### 1. 训练脚本 (train_encdec_7d.py)
- **数据处理**: 修改 `make_ds()` 函数支持双目标变量
- **模型架构**: 更新 `EncDec` 类，添加两个输出头：
  - `fc_energy`: 电量预测头
  - `fc_power`: 功率预测头
- **损失函数**: 使用联合损失 `loss = loss_energy + loss_power`
- **标准化器**: 分别保存电量和功率的标准化器：
  - `scaler_y_energy.pkl`
  - `scaler_y_power.pkl`
- **评估**: 分别计算电量和功率的 MAPE 指标

### 2. 场站微调 (fine_tune_station.py)
- **数据加载**: 支持双输出数据集构建
- **模型结构**: 与训练脚本保持一致的双输出架构
- **微调策略**: 冻结 Encoder，只微调 Decoder 和两个输出头
- **评估**: 分别评估电量和功率预测性能

### 3. 增量微调 (incremental_finetune.py)
- **缓存管理**: 支持双输出的历史数据缓存
- **增量学习**: 基于双输出模型进行增量更新
- **性能监控**: 跟踪电量和功率的预测精度变化

### 4. API 服务 (api_encdec_7d.py)
- **模型加载**: 加载双输出标准化器
- **预测接口**: 返回电量和功率的双重预测
- **响应格式**: 更新 API 响应包含两个预测值：
  ```json
  {
    "ts": "2024-01-01T00:00:00",
    "forward_total_active_energy_pred": 105.2,
    "total_active_power_pred": 420.8
  }
  ```

### 5. 测试脚本 (test_api_encdec_7d.py)
- **数据准备**: 使用 `vpp_meter.csv` 的实际数据
- **结果展示**: 显示电量和功率的预测统计信息
- **文件保存**: 将完整预测结果保存到 JSON 文件

## 技术细节

### 模型架构
```python
class EncDec(nn.Module):
    def __init__(self, d_enc, d_dec, hid=128, drop=.24):
        super().__init__()
        self.enc = nn.LSTM(d_enc, hid, batch_first=True)
        self.dec = nn.LSTM(d_dec, hid, batch_first=True)
        self.dp = nn.Dropout(drop)
        self.fc_energy = nn.Linear(hid, 1)  # 电量预测
        self.fc_power = nn.Linear(hid, 1)   # 功率预测
    
    def forward(self, xe, xd):
        _, (h, c) = self.enc(xe)
        out, _ = self.dec(xd, (h, c))
        out_dp = self.dp(out)
        energy_pred = self.fc_energy(out_dp).squeeze(-1)
        power_pred = self.fc_power(out_dp).squeeze(-1)
        return energy_pred, power_pred
```

### 损失函数
- 使用加权 L1 损失分别计算电量和功率损失
- 总损失 = 电量损失 + 功率损失
- 对第3-4天预测给予更高权重

### 数据流程
1. **输入**: 历史电量、功率及其他特征
2. **编码**: LSTM Encoder 提取时序特征
3. **解码**: LSTM Decoder 生成未来序列表示
4. **预测**: 两个独立的全连接层分别预测电量和功率
5. **输出**: 未来7天每15分钟的电量和功率预测

## 性能优势

1. **联合学习**: 电量和功率相关性强，联合训练提高预测精度
2. **共享特征**: Encoder-Decoder 共享，减少参数量
3. **独立输出**: 两个预测头独立，避免相互干扰
4. **一致性**: 电量和功率预测在时序模式上保持一致

## 使用方法

### 训练通用模型
```bash
python train_encdec_7d.py
```

### 场站微调
```bash
# 修改 STATION_ID 后运行
python fine_tune_station.py
```

### 增量更新
```bash
python incremental_finetune.py
```

### 启动API服务
```bash
python api_encdec_7d.py
```

### 测试API
```bash
python test_api_encdec_7d.py
```

## 输出文件

训练完成后会生成：
- `model_weighted.pth`: 通用双输出模型
- `scaler_y_energy.pkl`: 电量标准化器
- `scaler_y_power.pkl`: 功率标准化器
- `model_{STATION_ID}/model_optimized_{STATION_ID}.pth`: 场站专用模型

## 注意事项

1. 确保 `vpp_meter.csv` 包含 `total_active_power` 字段
2. 电量和功率的量纲差异较大，使用独立的标准化器
3. API 返回的功率预测是直接从模型输出，不是通过电量计算
4. 双输出模型的训练时间会略有增加，但预测精度显著提升