import pandas as pd
import numpy as np
import os

# ===================== 配置参数 =====================
# 输入：扩容后的路段级CSV文件路径
INPUT_CSV = "traffic_road_level_big.csv"
# 输出：与PEMS04格式一致且包含sumo_traffic键的NPZ文件路径
OUTPUT_NPZ = "sumo_traffic_pems_format.npz"
# PEMS04特征映射（保证维度对齐）
# SUMO特征 → PEMS04特征：flow(流量)、avg_speed(速度)、occupation(占有率，SUMO无则设为0)
FEATURE_MAP = {
    "flow": 0,        # 第0维：流量
    "avg_speed": 1,   # 第1维：平均速度
    "occupation": 2   # 第2维：占有率（占位，值为0）
}

# ===================== 核心转换函数 =====================
def csv_to_pems_npz(csv_path, npz_path):
    """
    将SUMO扩容后的CSV路段级数据转换为PEMS04格式的.npz文件
    关键修改：同时保存sumo_traffic和data键，兼容原迁移学习代码
    输出结构：
    - sumo_traffic: (时间步数量, 路段数量, 特征数量) → 适配原代码
    - data: 同上 → 兼容PEMS04
    - time_steps/edge_ids: 溯源信息
    """
    # 1. 加载CSV数据并校验
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"输入CSV文件不存在：{csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"加载CSV数据，共{len(df)}条记录")
    
    # 基础数据校验
    required_cols = ["time", "edge_id", "flow", "avg_speed"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV缺失必要列：{col}，请检查数据生成流程")
    
    # 2. 数据预处理：去重、排序、类型转换
    df = df.drop_duplicates(subset=["time", "edge_id"])  # 去重
    df = df.sort_values(by=["time", "edge_id"]).reset_index(drop=True)  # 按时间+路段排序
    df["time"] = df["time"].astype(int)
    df["edge_id"] = df["edge_id"].astype(str)
    df["flow"] = df["flow"].astype(float)
    df["avg_speed"] = df["avg_speed"].astype(float)
    
    # 3. 提取核心维度信息
    unique_times = sorted(df["time"].unique())  # 所有时间步（升序）
    unique_edges = sorted(df["edge_id"].unique())  # 所有路段ID（升序）
    time_steps = len(unique_times)  # 时间步数量
    edge_num = len(unique_edges)    # 路段数量
    feature_num = len(FEATURE_MAP)  # 特征数量（固定为3，与PEMS04一致）
    
    print(f"数据维度信息：")
    print(f"  - 时间步数量：{time_steps}")
    print(f"  - 路段数量：{edge_num}")
    print(f"  - 特征数量：{feature_num}（流量/速度/占有率）")
    
    # 4. 构建路段ID到索引的映射（便于重塑数组）
    edge_to_idx = {edge: idx for idx, edge in enumerate(unique_edges)}
    time_to_idx = {time: idx for idx, time in enumerate(unique_times)}
    
    # 5. 初始化三维数组（时间步×路段数×特征数）
    data = np.zeros((time_steps, edge_num, feature_num), dtype=np.float32)
    
    # 6. 填充数据到三维数组
    print("开始填充数据到三维数组...")
    for _, row in df.iterrows():
        # 获取当前记录的索引
        t_idx = time_to_idx[row["time"]]
        e_idx = edge_to_idx[row["edge_id"]]
        
        # 填充特征值
        data[t_idx, e_idx, FEATURE_MAP["flow"]] = row["flow"]          # 流量
        data[t_idx, e_idx, FEATURE_MAP["avg_speed"]] = row["avg_speed"]# 平均速度
        data[t_idx, e_idx, FEATURE_MAP["occupation"]] = 0.0            # 占有率（占位）
    
    # 7. 数据校验：确保无空值、无异常值
    if np.isnan(data).any():
        print("警告：数据中存在NaN值，已自动填充为0")
        data = np.nan_to_num(data, nan=0.0)
    
    if np.isinf(data).any():
        print("警告：数据中存在无穷值，已自动填充为0")
        data = np.nan_to_num(data, posinf=0.0, neginf=0.0)
    
    # 8. 保存为NPZ文件（核心修改：同时保存sumo_traffic和data键）
    np.savez_compressed(
        npz_path,
        sumo_traffic=data,         # 适配原迁移学习代码的键名
        data=data,                # 兼容PEMS04的键名
        time_steps=unique_times,  # 时间步列表（可选，便于溯源）
        edge_ids=unique_edges     # 路段ID列表（可选，便于溯源）
    )
    
    print(f"\n转换完成！生成兼容的NPZ文件：{npz_path}")
    print(f"NPZ文件包含键名：['sumo_traffic', 'data', 'time_steps', 'edge_ids']")
    print(f"核心数据形状：{data.shape} → (时间步, 路段数, 特征数)")
    return npz_path

# ===================== 验证转换结果 =====================
def verify_npz(npz_path):
    """验证生成的NPZ文件是否包含sumo_traffic键，且格式正确"""
    print("\n" + "="*50)
    print("验证NPZ文件格式（重点检查sumo_traffic键）...")
    print("="*50)
    
    npz_data = np.load(npz_path, allow_pickle=True)
    
    # 核心验证：sumo_traffic键是否存在
    if "sumo_traffic" not in npz_data.files:
        raise KeyError("NPZ文件中未生成sumo_traffic键！转换失败")
    else:
        print("✅ 验证通过：NPZ文件包含sumo_traffic键")
    
    # 读取核心数据
    sumo_traffic_data = npz_data["sumo_traffic"]
    data = npz_data["data"]
    time_steps = npz_data["time_steps"]
    edge_ids = npz_data["edge_ids"]
    
    print(f"NPZ文件结构验证：")
    print(f"  - sumo_traffic数据形状：{sumo_traffic_data.shape}（时间步×路段数×特征数）")
    print(f"  - data数据形状：{data.shape}（与sumo_traffic一致）")
    print(f"  - 时间步范围：{time_steps[0]} ~ {time_steps[-1]}")
    print(f"  - 前5个路段ID：{edge_ids[:5]}")
    print(f"  - 示例数据：第0时间步第0路段 → 流量={sumo_traffic_data[0,0,0]:.2f}，速度={sumo_traffic_data[0,0,1]:.2f}")
    
    # 校验数据类型（与PEMS04一致为float32）
    if sumo_traffic_data.dtype != np.float32:
        print(f"警告：数据类型为{sumo_traffic_data.dtype}，已自动转换为float32")
        sumo_traffic_data = sumo_traffic_data.astype(np.float32)
    
    print("✅ NPZ文件验证完全通过，可直接用于原迁移学习代码！")

# ===================== 主函数 =====================
if __name__ == "__main__":
    try:
        # 执行转换（生成含sumo_traffic键的NPZ）
        npz_file = csv_to_pems_npz(INPUT_CSV, OUTPUT_NPZ)
        
        # 验证结果（重点检查sumo_traffic键）
        verify_npz(npz_file)
        
    except Exception as e:
        print(f"转换失败：{e}")
        import traceback
        traceback.print_exc()