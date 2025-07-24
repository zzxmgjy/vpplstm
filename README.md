# 单站电力预测系统

该系统使用编码器-解码器LSTM模型为单个站点提供7天电力预测。系统以15分钟为间隔预测`total_active_power`（总有功功率）和`not_use_power`（不可控负荷）。

## 🏗️ 系统架构

```
数据输入 → 特征工程 → 模型训练 → API服务 → 预测结果
     ↓         ↓          ↓         ↓        ↓
CSV文件 → 时间/天气/电力 → 单站 → REST API → 7天预测
             特征        LSTM模型           (672个点)
```

## 📁 文件结构

```
├── train_encdec_7d_single_station.py    # 主训练脚本
├── incremental_finetune.py              # 微调脚本
├── api_encdec_7d_single_station.py      # API服务
├── test_api_encdec_7d_single_station.py # API测试脚本
├── run_training_example.py              # 训练示例运行器
├── merged_station_test.csv              # 输入数据文件
└── models/                              # 模型存储目录
    └── station_{id}/                    # 单站模型
        ├── model_power.pth              # 训练好的模型权重
        ├── model_power_finetuned.pth    # 微调后的模型（如果存在）
        ├── scaler_*.pkl                 # 数据缩放器
        ├── *_cols.pkl                   # 特征列
        ├── config.pkl                   # 模型配置
        └── station_info.pkl             # 站点元数据
```

## 🚀 快速开始

### 1. 训练初始模型

```bash
# 为特定站点训练模型
python train_encdec_7d_single_station.py --station_id YOUR_STATION_ID --data_file merged_station_test.csv

# 或使用示例脚本（训练找到的第一个站点）
python run_training_example.py
```

### 2. 启动API服务

```bash
python api_encdec_7d_single_station.py
```

API将在`http://localhost:5000`上启动

### 3. 测试API

```bash
python test_api_encdec_7d_single_station.py
```

### 4. 模型微调（可选）

```bash
python incremental_finetune.py --model_dir models/station_YOUR_ID --data_file new_data.csv --station_id YOUR_STATION_ID
```

## 📊 数据要求

### 输入数据格式

CSV文件应包含以下列：

**必需：**
- `ts`: 时间戳（ISO格式）
- `station_ref_id`: 站点标识符
- `total_active_power`: 总有功功率消耗

**可选：**
- `not_use_power`: 不可控负荷（如果缺失则自动生成）
- `temp`: 温度
- `humidity`: 湿度百分比
- `windSpeed`: 风速
- `code`: 天气代码

### 数据格式示例

```csv
ts,station_ref_id,total_active_power,not_use_power
2025-05-19 15:15:00,1716387625733984256,306.42,218.90
2025-05-19 15:20:00,1716387625733984256,322.56,213.70
2025-05-19 15:25:00,1716387625733984256,400.06,237.88
```

## 🔧 API端点

### 健康检查
```http
GET /health
```

### 列出可用站点
```http
GET /stations
```

### 加载站点模型
```http
POST /load_model/{station_id}
```

### 进行预测
```http
POST /predict/{station_id}
Content-Type: application/json

{
  "data": [
    {
      "ts": "2025-05-19T15:15:00",
      "total_active_power": 306.42,
      "not_use_power": 218.90,
      "temp": 25.0,
      "humidity": 60.0
    },
    ...
  ]
}
```

### 响应格式
```json
{
  "status": "success",
  "station_id": "1716387625733984256",
  "prediction_start": "2025-05-26T15:30:00",
  "prediction_end": "2025-06-02T15:15:00",
  "total_points": 672,
  "predictions": [
    {
      "timestamp": "2025-05-26T15:30:00",
      "total_active_power": 315.42,
      "not_use_power": 220.15,
      "step": 1,
      "day": 1,
      "hour_of_day": 15,
      "minute_of_hour": 30
    },
    ...
  ]
}
```

## 🧠 模型特征

### 输入特征（无能源字段）
- **时间特征**: 小时、分钟、星期几、月份、日期
- **日历特征**: 节假日、工作日、高峰时段
- **天气特征**: 温度、湿度、风速
- **电力滞后特征**: 1, 2, 4, 8, 12, 24, 48, 96步
- **滚动统计**: 移动平均值和标准差
- **周期性特征**: 时间的正弦/余弦变换
- **STL分解**: 趋势、季节性、残差成分

### 模型架构
- **编码器**: 用于历史数据处理的多层LSTM
- **解码器**: 用于未来预测的多层LSTM
- **双输出**: 同时预测两种电力指标
- **加权损失**: 对第3-4天预测给予更高权重

### 训练配置
```python
CFG = {
    'past_steps': 96*7,      # 7天历史数据（672个点）
    'future_steps': 96*7,    # 7天预测（672个点）
    'hidden_dim': 256,       # LSTM隐藏维度
    'num_layers': 2,         # LSTM层数
    'batch_size': 64,        # 训练批次大小
    'epochs': 300,           # 最大训练轮次
    'patience': 30,          # 早停耐心值
    'lr': 1e-4              # 学习率
}
```

## 🔄 工作流示例

### 完整训练工作流

```bash
# 1. 训练初始模型
python train_encdec_7d_single_station.py --station_id 1716387625733984256

# 2. 启动API服务
python api_encdec_7d_single_station.py &

# 3. 测试API
python test_api_encdec_7d_single_station.py

# 4. 使用新数据微调（如果有）
python incremental_finetune.py \
  --model_dir models/station_1716387625733984256 \
  --data_file new_incremental_data.csv \
  --station_id 1716387625733984256
```

### API使用示例

```python
import requests
import json

# 加载模型
response = requests.post('http://localhost:5000/load_model/1716387625733984256')

# 准备预测数据
data = {
    "data": [
        {
            "ts": "2025-05-19T15:15:00",
            "total_active_power": 306.42,
            "not_use_power": 218.90,
            "temp": 25.0,
            "humidity": 60.0
        }
        # ... 更多历史数据点（需要7天的672个点）
    ]
}

# 进行预测
response = requests.post(
    'http://localhost:5000/predict/1716387625733984256',
    json=data
)

predictions = response.json()
```

## 📈 性能指标

模型使用以下指标进行评估：
- **MAPE（平均绝对百分比误差）**: 按天和总体评估
- **加权L1损失**: 对关键预测天数给予更高权重
- **每日平均值**: 按天跟踪性能

## 🛠️ 故障排除

### 常见问题

1. **数据不足**
   - 训练至少需要672 + 672 = 1344个数据点
   - 确保数据以15分钟为间隔

2. **缺失列**
   - 系统会自动填充缺失的可选列
   - 检查日志中关于缺失特征的警告

3. **模型加载错误**
   - 确保模型目录存在
   - 检查训练是否成功完成

4. **API连接问题**
   - 验证API服务器是否在5000端口运行
   - 检查防火墙设置

### 调试命令

```bash
# 检查数据文件结构
head -5 merged_station_test.csv

# 列出可用模型
ls -la models/

# 检查API服务器状态
curl http://localhost:5000/health

# 查看训练日志
python train_encdec_7d_single_station.py --station_id YOUR_ID 2>&1 | tee training.log
```

## 🔮 未来增强

- [ ] 多站点批量训练
- [ ] 实时数据流
- [ ] 模型集成方法
- [ ] 高级天气集成
- [ ] 自动模型重训练
- [ ] 性能监控仪表板

## 📞 支持

如有问题或疑问：
1. 查看故障排除部分
2. 检查日志中的错误信息
3. 确保数据格式符合要求
4. 验证系统依赖项已安装

---

**注意**: 该系统专为单站电力预测设计。每个站点需要自己的训练模型以获得最佳性能。
