import pandas as pd
import numpy as np

# 1. 读取转换后的FCD CSV数据（确保fcd_data.csv在同一文件夹下）
df = pd.read_csv("fcd_data.csv")

# 2. 筛选核心列，去除无效数据
df = df[["time", "edge_id", "speed"]].dropna()
# 转换数据类型（避免聚合错误）
df["time"] = df["time"].astype(int)
df["edge_id"] = df["edge_id"].astype(str)

# 3. 聚合为路段级时序数据（每一秒每个路段的流量、平均速度）
traffic_data = df.groupby(["time", "edge_id"]).agg(
    flow=("speed", "count"),  # 流量=该路段该时间的车辆数
    avg_speed=("speed", "mean")  # 平均速度=该路段该时间所有车辆的速度均值
).reset_index()

# 4. 保存为LSTM预处理的原始数据
traffic_data.to_csv("traffic_road_level.csv", index=False)

print("路段级时序数据生成完成！")
print("数据预览：")
print(traffic_data.head(10))