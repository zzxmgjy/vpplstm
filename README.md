## 概述
初始化安装包:
pip install pandas scikit-learn tensorflow fastapi uvicorn holidays

api.py 调用预测接口
train_final.py 最终训练模型代码
output/finetune_model.py 增量微调训练模型

load.csv 初始训练数据
out/incremental_data.csv 增量训练数据

## api传参接口说明
{
  "station_ref_id": "STATION_001",
  "historical_data": [
    {
      "energy_date": "2025-07-10T08:00:00Z",
      "load_discharge_delta": 180.5,
      "temp": 28.2,
      "code": 100,
      "humidity": 55.1,
      "windSpeed": 11.3,
      "cloud": 20,
      "is_work": 1,
      "is_peak": 1
    },
    // ... 此处应有前五天的【历史】数据点...
  ],
  "future_data": [
    {
      "energy_date": "2025-07-15T00:00:00Z",
      "temp": 22.5,
      "code": 101,
      "humidity": 66,
      "windSpeed": 7.9,
      "cloud": 32,
      "is_work": 1,
      "is_peak": 0
    },
    {
      "energy_date": "2025-07-15T09:00:00Z",
      "temp": 26.8,
      "code": 100,
      "humidity": 60,
      "windSpeed": 8.5,
      "cloud": 15,
      "is_work": 1,
      "is_peak": 1
    },
    // ... 此处应有 未来一天的96 个更多【未来】数据点, 每个都包含 is_work 和 is_peak ...
  ]
}