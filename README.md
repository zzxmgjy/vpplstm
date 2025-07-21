# 负荷预测服务 README

## 概述  
本项目用于对站点未来 24h（96 点）负荷进行预测，并支持增量微调。  

## 1. 安装依赖
```bash 
gpu服务器安装
pip install pandas scikit-learn fastapi uvicorn holidays matplotlib 

#确定cuda版本然后安装pytorch
nvidia-smi
#去官网找，然后安装
https://pytorch.org/get-started/previous-versions/



## API 接口文档
1. 请求地址
POST /predict
2. 请求头
Content-Type: application/json
3. 请求体示例
{
  "station_id": 9001,          // int 
  "past_data": [                 // 连续 576 条历史 15min 样本（5 天,比如7-10号到07-15号所有15分钟数据）
    {
      "energy_date": "2023-08-01T00:00:00",
      "load_discharge_delta": 123.4,
      "temp": 29.6,
      "humidity": 65.2,
      "windSpeed": 2.8,
      "is_work": 1,              // 可省略，服务器自动推导
      "is_peak": 0               // 可省略
    },
    …
    {
      "energy_date": "2023-08-05T23:45:00",
      "load_discharge_delta": 141.2,
      "temp": 28.0,
      "humidity": 70.1,
      "windSpeed": 1.9
    }
  ],

  "future_external": [           // 连续 672 条未来 15min 外生特征（7 天）
    {
      "energy_date": "2023-08-06T00:00:00",
      "temp": 28.5,
      "humidity": 71.0,
      "windSpeed": 2.1,
      "is_work": 0,              // 可省略
      "is_peak": 0               // 可省略
    },
    …
    {
      "energy_date": "2023-08-12T23:45:00",
      "temp": 30.1,
      "humidity": 68.7,
      "windSpeed": 3.0
    }
  ]
}
字段说明

字段	必填	类型	说明
station_id	✔	string/int	请求预测的场站 ID
past_data	✔	array	长度必须 480，时间间隔 15 min，并且连续不缺口
├─ energy_date	✔	ISO-8601	时间戳；如 2023-08-01T00:15:00
├─ load_discharge_delta	✔	float	15 min 负荷 / 功率值
├─ temp / humidity / windSpeed	✔	float	同训练数据度量单位保持一致
├─ is_work	✖	0 / 1	是否工作日（缺省时由 API 依据节假日 & 周几推导）
└─ is_peak	✖	0 / 1	是否 高峰时段 08:30-17:30（缺省时由 API 自动推导）
future_external	✔	array	长度必须 672；字段同 past_data，但不含负荷值

3.返回示例:
{
  "station_id": "9001",
  "model_used": "station",             // "station" = 单站微调模型; "base" = 通用基础模型
  "predictions": [
    {
      "energy_date": "2023-08-06T00:00:00",
      "load_discharge_delta_pred": 135.72
    },
    {
      "energy_date": "2023-08-06T00:15:00",
      "load_discharge_delta_pred": 136.05
    },
    …
    {
      "energy_date": "2023-08-12T23:45:00",
      "load_discharge_delta_pred": 128.43
    }
  ]
}
