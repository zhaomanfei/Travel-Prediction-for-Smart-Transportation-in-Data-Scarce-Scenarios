import pandas as pd
import numpy as np

# 1. 读取数据（PEMS关键参数替换为你的实际值）
sumo_df = pd.read_csv("traffic_road_level_big.csv")
pems_mean = 89.2  # 替换为你的PEMS流量均值
pems_std = 67.5   # 替换为你的PEMS流量标准差
seq_len = 12      # 与PEMS模型一致
pred_len = 3      # 与PEMS模型一致
n_features = 1    # 仅用流量特征，与PEMS模型一致

# 2. 数据预处理（与PEMS清洗流程一致）
# 2.1 提取流量特征，按时间+路段排序
sumo_df = sumo_df[["time", "edge_id", "flow"]].dropna()
sumo_df = sumo_df.sort_values(by=["time", "edge_id"]).reset_index(drop=True)

# 2.2 构建「时间步×路段」的流量矩阵（对齐PEMS格式）
unique_times = sorted(sumo_df["time"].unique())
unique_edges = sorted(sumo_df["edge_id"].unique())
n_nodes = len(unique_edges)  # SUMO数据的路段数

# 初始化流量矩阵
flow_matrix = np.zeros((len(unique_times), n_nodes))
for t_idx, t in enumerate(unique_times):
    t_data = sumo_df[sumo_df["time"] == t]
    for e_idx, edge in enumerate(unique_edges):
        flow_val = t_data[t_data["edge_id"] == edge]["flow"].values
        flow_matrix[t_idx, e_idx] = flow_val[0] if len(flow_val) > 0 else 0

# 2.3 用PEMS的归一化参数归一化（关键，对齐数据分布）
flow_matrix_norm = (flow_matrix - pems_mean) / pems_std
# 避免除以0或异常值
flow_matrix_norm = np.clip(flow_matrix_norm, -5, 5)

# 2.4 构建时序序列（与PEMS模型训练时的序列构建逻辑一致）
def create_sequences(matrix, seq_len, pred_len):
    X, y = [], []
    for i in range(len(matrix) - seq_len - pred_len + 1):
        # 输入：前seq_len个时间步的所有路段数据
        X.append(matrix[i:i+seq_len])
        # 输出：后pred_len个时间步的所有路段数据
        y.append(matrix[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)

X_sumo, y_sumo = create_sequences(flow_matrix_norm, seq_len, pred_len)

# 2.5 调整输入形状（匹配LSTM模型输入，增加特征维度）
# 转换前：(样本数, seq_len, n_nodes)
# 转换后：(样本数, seq_len, n_nodes, n_features)（与PEMS模型输入一致）
X_sumo = X_sumo.reshape(X_sumo.shape[0], X_sumo.shape[1], X_sumo.shape[2], n_features)
y_sumo = y_sumo.reshape(y_sumo.shape[0], y_sumo.shape[1], y_sumo.shape[2], n_features)

# 2.6 划分训练/验证集（SUMO数据量充足，按8:2划分，不打乱时序）
train_split = int(0.8 * len(X_sumo))
X_sumo_train, X_sumo_val = X_sumo[:train_split], X_sumo[train_split:]
y_sumo_train, y_sumo_val = y_sumo[:train_split], y_sumo[train_split:]

print("SUMO数据格式转换完成！")
print(f"SUMO输入形状：{X_sumo_train.shape}（与PEMS模型输入一致）")
print(f"SUMO输出形状：{y_sumo_train.shape}（与PEMS模型输出一致）")