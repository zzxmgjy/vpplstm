import pandas as pd
import requests
import json
from datetime import datetime, timedelta

# 场站ID
station_id = 3205103743359  # 使用数值类型

# 读取vpp_meter.csv文件
df = pd.read_csv('vpp_meter.csv')

# 将ts列转换为datetime类型
df['ts'] = pd.to_datetime(df['ts'])

# 检查必需字段
required_fields = ['ts', 'total_active_power', 'forward_total_active_energy', 
                   'backward_total_active_energy', 'label', 'station_ref_id']
missing_fields = [f for f in required_fields if f not in df.columns]
if missing_fields:
    print(f"错误：缺少必需字段: {missing_fields}")
    exit(1)

# 筛选出指定场站的数据
filtered_df = df[df['station_ref_id'] == station_id].copy()

# 打印筛选后的数据条数，用于调试
print(f"场站 {station_id} 的数据条数: {len(filtered_df)}")

# 确保有足够的数据
if len(filtered_df) == 0:
    print("错误：没有找到符合条件的数据！")
    exit(1)

# 确保数据按时间排序
filtered_df = filtered_df.sort_values('ts')

# 检查数据是否有足够的记录 (至少需要480条用于预测)
if len(filtered_df) < 480:
    print(f"警告：数据不足480条，只有{len(filtered_df)}条")
    print("错误：API需要连续的480条记录")
    exit(1)

# 取最后的数据作为历史数据
# 为了确保API能够正确计算滞后值和移动平均值，我们提供更多的历史数据
past_data = filtered_df.tail(min(len(filtered_df), 1000))  # 最多取1000条

# 打印past_data的时间范围
if not past_data.empty:
    print(f"过去数据的时间范围: {past_data['ts'].min()} 到 {past_data['ts'].max()}")
    print(f"过去数据的条数: {len(past_data)}")

# 构建past_data部分的请求数据
past_data_list = []
for _, row in past_data.iterrows():
    past_item = {
        "ts": row['ts'].strftime("%Y-%m-%dT%H:%M:%S"),
        "forward_total_active_energy": float(row['forward_total_active_energy']),
        "total_active_power": float(row.get('total_active_power', 0)),
        # 处理可能缺失的字段
        "temp": float(row.get('temp', 25.0)),
        "humidity": float(row.get('humidity', 60.0)),
        "windSpeed": float(row.get('windSpeed', 5.0)),
        "is_work": int(row.get('is_work', 1)),
        "is_peak": int(row.get('is_peak', 0)),
        "code": int(row.get('code', 999))
    }
    past_data_list.append(past_item)

# 获取最后一条数据的时间
last_time = past_data['ts'].iloc[-1]

# 构建future_external部分的请求数据（未来7天，672条记录）
future_data_list = []
for i in range(1, 673):
    # 每15分钟一条记录
    future_time = last_time + timedelta(minutes=15 * i)
    
    # 生成模拟的天气数据
    hour = future_time.hour
    temp = 25 + (hour % 12) * 0.8 if hour < 12 else 35 - (hour % 12) * 0.8
    humidity = 90 - (temp - 25) * 2
    windSpeed = (hour % 16)
    
    # 工作日判断（周一到周五为工作日）
    is_work = 1 if future_time.weekday() < 5 else 0
    
    # 高峰时段判断（8:30-17:30为高峰时段）
    is_peak = 1 if 8 <= hour < 18 else 0
    
    future_item = {
        "ts": future_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "temp": float(temp),
        "humidity": float(humidity),
        "windSpeed": float(windSpeed),
        "is_work": is_work,
        "is_peak": is_peak,
        "code": 999
    }
    future_data_list.append(future_item)

# 构建完整的请求数据
request_data = {
    "station_id": str(station_id),
    "past_data": past_data_list,
    "future_external": future_data_list
}

# 打印请求数据的长度信息，确认是否符合要求
print(f"过去数据条数: {len(past_data_list)}")
print(f"未来数据条数: {len(future_data_list)}")

# 发送请求到API
url = "http://localhost:8000/predict"
headers = {"Content-Type": "application/json"}

try:
    # 发送POST请求
    print("正在发送请求到API...")
    response = requests.post(url, json=request_data, headers=headers)
    
    # 检查响应状态码
    if response.status_code == 200:
        # 解析JSON响应
        result = response.json()
        
        # 打印响应结果摘要
        print("\n✅ 预测成功!")
        print(f"场站ID: {result['station_id']}")
        print(f"使用模型: {result['model_used']}")
        print(f"预测结果条数: {len(result['predictions'])}")
        
        # 打印前几条预测结果作为示例
        print("\n📊 预测结果示例 (前5条):")
        for i, pred in enumerate(result['predictions'][:5]):
            print(f"  {i+1}. 时间: {pred['ts']}")
            print(f"     用电量预测: {pred['forward_total_active_energy_pred']:.2f}")
            print(f"     功率预测: {pred['total_active_power_pred']:.2f}")
        
        # 保存完整结果到文件
        with open('prediction_results.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print("\n💾 完整预测结果已保存到 prediction_results.json")
        
        # 统计信息
        predictions = result['predictions']
        energy_preds = [p['forward_total_active_energy_pred'] for p in predictions]
        power_preds = [p['total_active_power_pred'] for p in predictions]
        
        print(f"\n📈 预测统计:")
        print(f"用电量预测范围: {min(energy_preds):.2f} ~ {max(energy_preds):.2f}")
        print(f"功率预测范围: {min(power_preds):.2f} ~ {max(power_preds):.2f}")
        print(f"平均用电量预测: {sum(energy_preds)/len(energy_preds):.2f}")
        print(f"平均功率预测: {sum(power_preds)/len(power_preds):.2f}")
        
    else:
        print(f"❌ 请求失败，状态码: {response.status_code}")
        print(f"错误信息: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("❌ 连接错误：无法连接到API服务器")
    print("请确保API服务器正在运行 (python api_encdec_7d.py)")
except Exception as e:
    print(f"❌ 发生错误: {str(e)}")