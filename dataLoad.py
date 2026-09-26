import os
import numpy as np
import pandas as pd
from scipy.io import loadmat

# 已手动解压的路径（保持不变）
extract_path = "D:/highSchool/科研/paper/LSTM/数据/TSdatasets/iTransformer_datasets/"

# ---------------------- 辅助排查：打印目录结构（保留，方便确认）----------------------
def print_folder_structure(root_path):
    print(f"\n当前查找根路径：{root_path}")
    print("路径下的所有文件夹和文件：")
    for root, dirs, files in os.walk(root_path):
        relative_path = os.path.relpath(root, root_path)
        print(f"\n--- {relative_path if relative_path != '.' else '根目录'} ---")
        for dir in dirs:
            print(f"  [文件夹] {dir}")
        for file in files:
            print(f"  [文件] {file}")

# 先打印目录结构（可选，确认文件位置）
print_folder_structure(extract_path)

# ---------------------- 加载traffic数据集（新增.xlsx格式支持，适配实际文件）----------------------
def load_traffic_data(data_path):
    # 实际的traffic文件路径（根据目录结构：traffic/traffic.xlsx）
    traffic_file = os.path.join(data_path, "traffic", "traffic.xlsx")
    
    if os.path.exists(traffic_file):
        print(f"\n找到traffic文件：{traffic_file}")
        # 加载Excel文件（pandas支持.xlsx格式）
        df = pd.read_excel(traffic_file)
        # 自动判断是否跳过第一列（时间列）
        if pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]) or isinstance(df.iloc[0, 0], str):
            data = df.values[:, 1:]  # 跳过时间列（如果第一列是时间/字符串）
        else:
            data = df.values  # 无时间列，全部加载
        
        print(f"traffic数据集形状：{data.shape}")
        return data
    else:
        print(f"\nERROR：未找到traffic文件！实际路径应为：{traffic_file}")
        print("请检查traffic文件夹下是否存在 traffic.xlsx 文件")
        return None

# ---------------------- 加载pems数据集（适配实际的PEMS文件夹和.npz格式）----------------------
def load_pems_data(data_path):
    # 实际的PEMS文件路径（文件夹名是PEMS，大写！文件是PEMS04.npz等）
    pems_file = os.path.join(data_path, "PEMS", "PEMS04.npz")  # 选PEMS04.npz，也可换PEMS03/07/08
    
    if os.path.exists(pems_file):
        print(f"\n找到pems文件：{pems_file}")
        # 加载.npz文件
        npz_data = np.load(pems_file)
        print(f"npz文件中的变量名：{list(npz_data.keys())}")  # 打印变量名，确认数据所在
    
        # 多数iTransformer的PEMS数据中，变量名是'data'（若不是，根据打印结果修改）
        data = npz_data['data'] if 'data' in npz_data else npz_data[list(npz_data.keys())[0]]
        
        # 调整形状为 (时间步, 传感器数)（PEMS数据通常是 (传感器数, 时间步)，需转置）
        if data.shape[0] < data.shape[1]:  # 传感器数 < 时间步 → 转置
            data = data.T
        
        print(f"pems数据集形状：{data.shape}")
        return data
    else:
        print(f"\nERROR：未找到pems文件！实际路径应为：{pems_file}")
        print("请检查PEMS文件夹下是否存在 PEMS04.npz（或PEMS03/07/08.npz）")
        return None

# ---------------------- 主程序：加载数据（二选一，按需注释）----------------------
if __name__ == "__main__":
    # 选项1：加载traffic数据集（Excel格式，已适配）
    data = load_traffic_data(extract_path)
    
    # 选项2：加载pems数据集（NPZ格式，已适配PEMS文件夹）
    # data = load_pems_data(extract_path)
    
    if data is not None:
        print("\n✅ 数据集加载成功！")
        print(f"数据集维度：时间步={data.shape[0]}, 传感器数/特征数={data.shape[1]}")
        # 后续可直接用于数据预处理（如归一化、构造LSTM输入序列）
    else:
        print("\n❌ 数据集加载失败，请根据上述提示检查文件！")