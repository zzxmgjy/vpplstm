# 负荷预测服务 README

## 概述  
本项目用于对站点未来 7 天（672 点，每 15 分钟一个点）的用电量和功率进行预测，支持通用模型训练、单站微调和增量更新。

## 工作流程
1. **通用模型训练** (`train_encdec_7d.py`) - 使用 vpp_meter.csv 训练基础 LSTM 模型
2. **单站微调** (`fine_tune_station.py`) - 基于通用模型为特定场站生成微调模型  
3. **增量微调** (`incremental_finetune.py`) - 使用新数据增量更新场站模型
4. **API 服务** (`api_encdec_7d.py`) - 提供预测接口，自动选择最优模型

## 数据字段要求
- **必需字段**: ts, total_active_power, forward_total_active_energy, backward_total_active_energy, label, station_ref_id
- **可选字段**: 天气信息（temp, humidity, windSpeed）、工作日标识、高峰时段标识等
- **输出**: 双输出预测（用电量 + 功率），未来 7 天每 15 分钟一个预测点

## 1. 安装依赖
```bash 
gpu服务器安装
pip install fastapi uvicorn pydantic pandas numpy scikit-learn joblib holidays  statsmodels lightgbm

#确定cuda版本然后安装pytorch
nvidia-smi
#去官网找，然后安装
https://pytorch.org/get-started/previous-versions/



## API 接口文档

### 1. 请求地址
```
POST /predict
```

### 2. 请求头
```
Content-Type: application/json
```

### 3. 请求体示例
```json
{
  "station_id": "1851144626925211648",
  "past_data": [
    {
      "ts": "2023-08-01T00:00:00",
      "forward_total_active_energy": 123.4,
      "total_active_power": 50.2,
      "temp": 29.6,
      "humidity": 65.2,
      "windSpeed": 2.8,
      "is_work": 1,
      "is_peak": 0,
      "code": 999
    }
  ],
  "future_external": [
    {
      "ts": "2023-08-06T00:00:00",
      "temp": 28.5,
      "humidity": 71.0,
      "windSpeed": 2.1,
      "is_work": 0,
      "is_peak": 0,
      "code": 999
    }
  ]
}
```

### 4. 字段说明

| 字段 | 必填 | 类型 | 说明 |
|------|------|------|------|
| station_id | ✔ | string | 请求预测的场站 ID |
| past_data | ✔ | array | 长度必须 ≥480，时间间隔 15 min，连续不缺口 |
| ├─ ts | ✔ | ISO-8601 | 时间戳，如 2023-08-01T00:15:00 |
| ├─ forward_total_active_energy | ✔ | float | 15 min 用电量值（负荷电量） |
| ├─ total_active_power | ✖ | float | 总有功功率 |
| ├─ temp | ✖ | float | 温度，默认 25.0 |
| ├─ humidity | ✖ | float | 湿度，默认 60.0 |
| ├─ windSpeed | ✖ | float | 风速，默认 5.0 |
| ├─ is_work | ✖ | 0/1 | 是否工作日（缺省时由 API 自动推导） |
| ├─ is_peak | ✖ | 0/1 | 是否高峰时段 08:30-17:30（缺省时由 API 自动推导） |
| └─ code | ✖ | int | 天气代码，默认 999 |
| future_external | ✔ | array | 长度必须 672（7天×96点），未来外生特征 |
| ├─ ts | ✔ | ISO-8601 | 未来时间戳 |
| ├─ temp/humidity/windSpeed | ✖ | float | 未来天气信息 |
| ├─ is_work/is_peak | ✖ | 0/1 | 未来工作日和高峰时段标识 |
| └─ code | ✖ | int | 未来天气代码 |

### 5. 返回示例
```json
{
  "station_id": "1851144626925211648",
  "model_used": "station",
  "predictions": [
    {
      "ts": "2023-08-06T00:00:00",
      "forward_total_active_energy_pred": 135.72,
      "total_active_power_pred": 51.25
    },
    {
      "ts": "2023-08-06T00:15:00",
      "forward_total_active_energy_pred": 136.05,
      "total_active_power_pred": 52.10
    }
  ]
}
```

### 6. 返回字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| station_id | string | 场站 ID |
| model_used | string | 使用的模型类型："station"=单站微调模型，"base"=通用基础模型 |
| predictions | array | 预测结果数组，包含672个15分钟间隔的预测点 |
| ├─ ts | ISO-8601 | 预测时间点 |
| ├─ forward_total_active_energy_pred | float | 预测的用电量 |
| └─ total_active_power_pred | float | 预测的功率 |

### 7. 错误响应
```json
{
  "detail": "past_data 需 ≥480 行"
}
```
