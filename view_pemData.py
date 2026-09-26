import numpy as np
import pandas as pd

# 1. 加载 .npz 文件并降维（处理 3维→2维）
with np.load(r"D:\highSchool\科研\paper\LSTM\Pro\LSTMTrafficPro\PEMS03.npz") as npz_file:
    data = npz_file["data"]  # 原始形状：(26208, 358, 1)
    data_2d = data.squeeze()  # 降维后：(26208, 358)
    print(f"数据形状：{data_2d.shape}（时间步 × 传感器数量）")

# 2. 创建 pandas 表格（适配交通数据）
# 列名：传感器1~传感器358（贴合 PEMS03 数据集）
# 行名：时间步1~时间步26208（清晰标识时序）
df = pd.DataFrame(
    data_2d,
    columns=[f"传感器{col+1}" for col in range(data_2d.shape[1])],
    index=[f"时间步{step+1}" for step in range(data_2d.shape[0])]
)

# 3. 数据预览（终端打印前10行×前5列，避免刷屏）
print("\n数据预览（前10时间步 × 前5传感器）：")
print(df.iloc[:10, :5])

# 4. 交互式可视化（弹出独立表格窗口，支持滚动/排序/筛选）
# 只显示前 5000 行、前 30 列（避免数据量过大导致卡顿）
df_view = df.iloc[:5000, :30]
df_view.head()  # 触发交互式窗口

# 导出前1000行、前50列到 Excel（避免文件过大）
df.iloc[:1000, :50].to_excel("PEMS03_交通数据_预览.xlsx", index=True)
print("已导出部分数据到 Excel 文件！")