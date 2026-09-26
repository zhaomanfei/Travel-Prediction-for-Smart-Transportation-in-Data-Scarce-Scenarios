import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 配置参数
DATA_LENGTH = 2000  # 数据条数（5分钟粒度，约7天数据）
INTERVAL_MINUTES = 5  # 时间间隔（分钟）
PEAK_HOUR1_START = 8  # 早高峰开始时间（时）
PEAK_HOUR1_END = 10   # 早高峰结束时间（时）
PEAK_HOUR2_START = 17 # 晚高峰开始时间（时）
PEAK_HOUR2_END = 19   # 晚高峰结束时间（时）

# 生成时间序列
start_time = datetime.now() - timedelta(minutes=INTERVAL_MINUTES*DATA_LENGTH)
time_list = [start_time + timedelta(minutes=INTERVAL_MINUTES*i) for i in range(DATA_LENGTH)]

# 生成基础车速（正态分布）
base_speed = np.random.normal(40, 8, DATA_LENGTH)
# 加入高峰时段车速降低
speed_adjust = np.ones(DATA_LENGTH)
for i, t in enumerate(time_list):
    hour = t.hour
    if (PEAK_HOUR1_START <= hour <= PEAK_HOUR1_END) or (PEAK_HOUR2_START <= hour <= PEAK_HOUR2_END):
        speed_adjust[i] = 0.5  # 高峰时段车速减半
    elif 0 <= hour <= 6:
        speed_adjust[i] = 1.2  # 凌晨车速更快
# 最终车速
final_speed = base_speed * speed_adjust
final_speed = np.clip(final_speed, 10, 80)  # 限制车速范围

# 计算拥堵等级（0-畅通,1-缓行,2-拥堵,3-严重拥堵）
congestion_level = np.zeros(DATA_LENGTH, dtype=int)
congestion_level[(final_speed >= 50)] = 0
congestion_level[(final_speed >= 30) & (final_speed < 50)] = 1
congestion_level[(final_speed >= 15) & (final_speed < 30)] = 2
congestion_level[(final_speed < 15)] = 3

# 构建DataFrame
df = pd.DataFrame({
    "timestamp": [t.strftime("%Y-%m-%d %H:%M:%S") for t in time_list],
    "speed_kmh": final_speed.round(2),
    "congestion_level": congestion_level,
    "road_id": "RD001"  # 模拟路段ID
})

# 保存为CSV（可直接用于LSTM训练）
df.to_csv("simulated_traffic_data.csv", index=False, encoding="utf-8-sig")
print(f"模拟交通数据已生成：{DATA_LENGTH}条，保存至simulated_traffic_data.csv")
print("数据预览：")
print(df.head(10))