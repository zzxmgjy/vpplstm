import pandas as pd
import requests
import json
from datetime import datetime, timedelta

# 场站ID
station_id = 1716387625733984256  # 改为数值类型，而不是字符串

# 读取loaddata.csv文件
df = pd.read_csv('loaddata.csv')

# 将energy_date列转换为datetime类型
df['energy_date'] = pd.to_datetime(df['energy_date'])

# 筛选出2025-07-10到2025-07-15的数据作为过去5天的数据
# 注意：API需要连续的480条记录（5天，每15分钟一条）
start_date = pd.to_datetime('2025-07-10 00:00:00')
end_date = pd.to_datetime('2025-07-15 23:45:00')
filtered_df = df[(df['energy_date'] >= start_date) & (df['energy_date'] <= end_date)]

# 筛选出station_ref_id为指定场站ID的数据
filtered_df = filtered_df[filtered_df['station_ref_id'] == station_id]

# 打印筛选后的数据条数，用于调试
print(f"筛选后的数据条数: {len(filtered_df)}")

# 确保有足够的数据
if len(filtered_df) == 0:
    print("错误：没有找到符合条件的数据！")
    exit(1)

# 确保数据按时间排序
filtered_df = filtered_df.sort_values('energy_date')

# 确保数据是连续的
# 检查数据是否有足够的记录
if len(filtered_df) < 480:
    print(f"警告：数据不足480条，只有{len(filtered_df)}条")
    print("错误：API需要连续的480条记录")
    exit(1)

# 确保数据是按15分钟间隔的连续数据
# 创建一个完整的时间索引（从开始到结束，每15分钟一个点）
full_date_range = pd.date_range(start=filtered_df['energy_date'].min(), 
                              end=filtered_df['energy_date'].max(), 
                              freq='15min')

# 检查是否有缺失的时间点
missing_dates = set(full_date_range) - set(filtered_df['energy_date'])
if missing_dates:
    print(f"警告：数据中有{len(missing_dates)}个缺失的时间点")
    # 如果有缺失的时间点，我们可以尝试插值或者其他方法填充
    # 这里简单起见，我们只取最后的连续480条记录
    # 首先按时间排序
    filtered_df = filtered_df.sort_values('energy_date')
    
    # 找出最长的连续段
    # 这里简化处理，直接取最后480条
    past_data = filtered_df.tail(480)
else:
    print("数据是连续的，没有缺失的时间点")
    # 取最后480条记录
    filtered_df = filtered_df.sort_values('energy_date')
    past_data = filtered_df.tail(480)

# 检查数据是否足够长，能够计算滞后值和移动平均值
# 根据API代码，它需要计算load_lag96，这需要至少96个历史点
# 我们需要确保数据至少有96+480=576个点
if len(filtered_df) < 576:
    print(f"警告：数据不足以计算滞后值和移动平均值，需要至少576条，但只有{len(filtered_df)}条")
    # 这里我们可以尝试生成一些模拟数据来填充
    # 但为了简单起见，我们直接使用现有数据
    # 注意：这可能会导致API返回错误

# 为了确保API能够正确计算滞后值和移动平均值，我们需要提供足够长的历史数据
# 这里我们提供所有可用的数据，而不仅仅是最后480条
past_data = filtered_df

# 打印past_data的时间范围
if not past_data.empty:
    print(f"过去数据的时间范围: {past_data['energy_date'].min()} 到 {past_data['energy_date'].max()}")
    print(f"过去数据的条数: {len(past_data)}")

# 构建past_data部分的请求数据
past_data_list = []
for _, row in past_data.iterrows():
    past_item = {
        "energy_date": row['energy_date'].strftime("%Y-%m-%dT%H:%M:%S"),
        "load_discharge_delta": float(row['load_discharge_delta']),
        "temp": float(row['temp']),
        "humidity": float(row['humidity']),
        "windSpeed": float(row['windSpeed']),
        "is_work": int(row['is_work']),
        "is_peak": int(row['is_peak'])
    }
    past_data_list.append(past_item)

# 获取最后一条数据的时间
last_time = past_data['energy_date'].iloc[-1]

# 构建future_external部分的请求数据（未来7天，672条记录）
future_data_list = []
for i in range(1, 673):
    # 每15分钟一条记录
    future_time = last_time + timedelta(minutes=15 * i)
    
    # 生成模拟的天气数据（这里简单地使用一些固定值，实际应用中可能需要更复杂的模型）
    # 温度在25-35度之间波动
    hour = future_time.hour
    temp = 25 + (hour % 12) * 0.8 if hour < 12 else 35 - (hour % 12) * 0.8
    
    # 湿度在50-90%之间波动，与温度成反比
    humidity = 90 - (temp - 25) * 2
    
    # 风速在0-15之间波动
    windSpeed = (hour % 16)
    
    # 工作日判断（周一到周五为工作日）
    is_work = 1 if future_time.weekday() < 5 else 0
    
    # 高峰时段判断（8:30-17:30为高峰时段）
    is_peak = 1 if 8 <= hour < 18 else 0
    
    future_item = {
        "energy_date": future_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "temp": float(temp),
        "humidity": float(humidity),
        "windSpeed": float(windSpeed),
        "is_work": is_work,
        "is_peak": is_peak
    }
    future_data_list.append(future_item)

# 构建完整的请求数据
request_data = {
    "station_id": str(station_id),  # 确保API请求中是字符串格式
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
    response = requests.post(url, json=request_data, headers=headers)
    
    # 检查响应状态码
    if response.status_code == 200:
        # 解析JSON响应
        result = response.json()
        
        # 打印响应结果
        print("\n预测结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # 打印预测结果的条数
        print(f"\n预测结果条数: {len(result['predictions'])}")
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print(f"错误信息: {response.text}")
        
except Exception as e:
    print(f"发生错误: {str(e)}")