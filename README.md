# 负荷预测服务 README

## 概述  
本项目用于对站点未来 24h（96 点）负荷进行预测，并支持增量微调。  

## 1. 安装依赖
```bash 
pip install pandas scikit-learn tensorflow fastapi uvicorn holidays matplotlib

## 2. 目录结构
.
├── api.py                 # FastAPI 预测接口
├── train_final.py         # 全量训练脚本
├── output/
│   └── finetune_model.py  # 增量微调脚本
├── load.csv               # 初始训练数据
└── out/
    └── incremental_data.csv  # 增量训练数据

## API 接口文档
1. 请求地址
POST /predict
2. 请求头
Content-Type: application/json
3. 请求体示例
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
    }
    /* 需补充前 5 天历史数据，共 5×96=480 条 */
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
    }
    /* 需补充未来 1 天数据，共 96 条 */
  ]
}
| 字段名                    | 类型     | 说明                 |
| ---------------------- | ------ | ------------------ |
| station\_ref\_id       | string | 站点唯一标识             |
| historical\_data       | array  | 过去 5 天历史数据（480 条）  |
| future\_data           | array  | 预测日未来 24h 数据（96 条） |
| energy\_date           | string | ISO-8601 时间戳（UTC）  |
| load\_discharge\_delta | float  | 负荷-放电差值，仅历史数据有     |
| temp                   | float  | 温度（°C）             |
| code                   | int    | 天气编码               |
| humidity               | float  | 湿度（%）              |
| windSpeed              | float  | 风速（m/s）            |
| cloud                  | int    | 云量（%）              |
| is\_work               | int    | 是否工作日（0/1）         |
| is\_peak               | int    | 是否峰时段（0/1）         |

3.返回示例:
{
  "station_ref_id": "STATION_001",
  "predictions": [
    {
      "energy_date": "2025-07-15T00:00:00Z",
      "load": 150.2
    },
    ...
  ]
}
