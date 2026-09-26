import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler

# ---------------------- 第一步：先加载数据集（复用之前的加载逻辑）----------------------
def load_traffic_data(data_path):
    """加载traffic数据集（Excel格式）"""
    traffic_file = os.path.join(data_path, "traffic", "traffic.xlsx")
    if os.path.exists(traffic_file):
        df = pd.read_excel(traffic_file)
        # 跳过时间列（如果第一列是时间/字符串）
        if pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]) or isinstance(df.iloc[0, 0], str):
            data = df.values[:, 1:].astype(np.float32)  # 转为float32，减少内存占用
        else:
            data = df.values.astype(np.float32)
        print(f"✅ traffic数据集加载成功，形状：{data.shape}")
        return data
    else:
        print(f"❌ 未找到traffic文件：{traffic_file}")
        return None

def load_pems_data(data_path):
    """加载PEMS数据集（NPZ格式）"""
    pems_file = os.path.join(data_path, "PEMS", "PEMS04.npz")
    if os.path.exists(pems_file):
        npz_data = np.load(pems_file)
        data = npz_data['data'].astype(np.float32)
        # 调整为 (时间步, 传感器数)
        if data.shape[0] < data.shape[1]:
            data = data.T
        print(f"✅ PEMS数据集加载成功，形状：{data.shape}")
        return data
    else:
        print(f"❌ 未找到PEMS文件：{pems_file}")
        return None

# ---------------------- 第二步：数据预处理核心函数 ----------------------
def data_preprocess(data, seq_len=12, pred_len=3):
    """
    数据预处理：归一化 + 构造LSTM输入序列
    参数：
        data: 原始数据 (time_steps, num_sensors)
        seq_len: 输入序列长度（用过去多少个时间步预测，默认12=60分钟）
        pred_len: 预测长度（预测未来多少个时间步，默认3=15分钟）
    返回：
        x_train, y_train: 训练集 (样本数, seq_len, num_sensors) / (样本数, pred_len, num_sensors)
        x_test, y_test: 测试集（同上）
        scaler: 归一化器（用于后续反归一化）
    """
    # 1. 归一化（对每个传感器单独归一化，避免不同量级影响）
    num_sensors = data.shape[1]
    scaler = MinMaxScaler(feature_range=(0, 1))  # 缩放到[0,1]
    data_scaled = scaler.fit_transform(data)  # shape: (time_steps, num_sensors)
    print(f"✅ 数据归一化完成，归一化后形状：{data_scaled.shape}")

    # 2. 构造LSTM输入输出序列（滑窗法）
    x, y = [], []
    for i in range(len(data_scaled) - seq_len - pred_len + 1):
        # 输入：过去seq_len个时间步的所有传感器数据
        x.append(data_scaled[i:i+seq_len, :])
        # 输出：未来pred_len个时间步的所有传感器数据
        y.append(data_scaled[i+seq_len:i+seq_len+pred_len, :])
    
    x = np.array(x)  # shape: (样本数, seq_len, num_sensors)
    y = np.array(y)  # shape: (样本数, pred_len, num_sensors)
    print(f"✅ 序列构造完成，输入形状：{x.shape}，输出形状：{y.shape}")

    # 3. 划分训练集和测试集（默认8:2分割）
    train_size = int(0.8 * len(x))
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print(f"✅ 数据集划分完成：")
    print(f"  - 训练集：x_train={x_train.shape}, y_train={y_train.shape}")
    print(f"  - 测试集：x_test={x_test.shape}, y_test={y_test.shape}")

    return x_train, y_train, x_test, y_test, scaler

# ---------------------- 主程序：执行加载+预处理 ----------------------
if __name__ == "__main__":
    # 数据集根路径（确保和你之前的路径一致）
    extract_path = "D:/highSchool/科研/paper/LSTM/数据/TSdatasets/iTransformer_datasets/"

    # 选择加载的数据集（二选一，注释另一个）
    data = load_traffic_data(extract_path)  # 加载traffic（Excel）
    # data = load_pems_data(extract_path)  # 加载PEMS（NPZ）

    # 若数据加载成功，执行预处理
    if data is not None:
        # 可调整seq_len（输入序列长度）和pred_len（预测长度）
        # 例如：seq_len=24（2小时），pred_len=6（30分钟）
        x_train, y_train, x_test, y_test, scaler = data_preprocess(
            data, seq_len=12, pred_len=3
        )

        # （可选）保存预处理后的数据，方便后续模型训练直接加载
        save_path = "D:/highSchool/科研/paper/LSTM/Pro/LSTMTrafficPro/preprocessed_data/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        np.save(os.path.join(save_path, "x_train.npy"), x_train)
        np.save(os.path.join(save_path, "y_train.npy"), y_train)
        np.save(os.path.join(save_path, "x_test.npy"), x_test)
        np.save(os.path.join(save_path, "y_test.npy"), y_test)
        # 保存scaler（后续预测时需要反归一化）
        import joblib
        joblib.dump(scaler, os.path.join(save_path, "scaler.pkl"))

        print(f"\n✅ 预处理数据已保存至：{save_path}")
        print("后续模型训练可直接加载这些.npy文件，无需重复预处理！")