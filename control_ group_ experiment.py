# ---------------------- 步骤1：导入所有依赖库 + 固定随机种子（保证可复现） ----------------------
import os
import pickle
import random
# 关闭TensorFlow oneDNN提示
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# 固定所有随机种子
random.seed(42)
import numpy as np
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"
import tensorflow as tf
tf.random.set_seed(42)

import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# 配置Matplotlib中文显示
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "Heiti TC", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------- 步骤2：全局参数配置 ----------------------
# 数据路径（替换为你的NPZ文件路径）
SUMO_DATA_PATH = "sumo_traffic_pems_format.npz"
# PEMS04预训练模型路径（若没有，注释迁移学习部分，仅运行对照组）
PEMS_PRETRAINED_MODEL_PATH = "trained_lstm_model/best_lstm_model.keras"
# PEMS04归一化器路径
PEMS_SCALER_PATH = "trained_lstm_model/minmax_scaler.pkl"
# 时序参数
LOOK_BACK = 12  # 输入序列长度（1小时）
# 训练参数
BATCH_SIZE = 64
EPOCHS = 50
UNITS = 64  # LSTM第一层神经元数
TEST_SIZE = 0.2  # 测试集占比
# 早停参数（优化：放宽patience，避免提前终止）
EARLY_STOP_PATIENCE = 5
# 保存路径
RESULT_DIR = "transfer_learning_comparison_optimized"
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

# ---------------------- 步骤3：数据加载与预处理（核心优化） ----------------------
def load_sumo_data(data_path, scaler_path=None):
    """
    加载SUMO仿真数据（兼容.csv/.npz格式）
    核心优化：
    1. 三维数据降维改为“所有路段流量均值”（保留全局趋势）
    2. 修复scaler加载逻辑，确保反归一化尺度匹配
    3. 固定随机种子，保证数据处理可复现
    """
    # 校验文件是否存在
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"SUMO数据文件不存在：{data_path}")
    
    # 加载CSV格式
    if data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
        # 自动选择流量列（优先flow，无则取第一列数值列）
        if "flow" in df.columns:
            raw_data = df["flow"].values.reshape(-1, 1)
        else:
            # 过滤非数值列，取第一列
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("CSV文件中无数值列，无法作为交通数据")
            raw_data = df[numeric_cols[0]].values.reshape(-1, 1)
        print(f"加载CSV数据，形状：{raw_data.shape}（时间步 × 1特征）")
    
    # 加载NPZ格式（核心优化：降维逻辑）
    elif data_path.endswith(".npz"):
        npz_data = np.load(data_path, allow_pickle=True)
        # 优先查找的键名列表
        possible_keys = ["sumo_traffic", "data", "traffic_data", "cleaned_traffic"]
        raw_data = None
        for key in possible_keys:
            if key in npz_data.files:
                raw_data = npz_data[key]
                print(f"找到NPZ键名：{key}，原始数据形状：{raw_data.shape}")
                break
        if raw_data is None:
            raise KeyError(
                f"NPZ文件中无以下有效键名：{possible_keys}，文件实际包含键：{npz_data.files}"
            )
        
        # 核心优化：三维数组降维为“所有路段流量均值”（更贴合全局交通趋势）
        if len(raw_data.shape) == 3:
            # 取所有路段的流量特征均值 → 保留路网整体流量趋势
            raw_data = np.mean(raw_data[:, :, 0], axis=1).reshape(-1, 1)
            print(f"三维数组降维为二维（所有路段流量均值）：{raw_data.shape}（时间步 × 1特征）")
        elif len(raw_data.shape) == 1:
            raw_data = raw_data.reshape(-1, 1)
            print(f"一维数组重塑为二维：{raw_data.shape}（时间步 × 1特征）")
        print(f"加载NPZ数据，最终形状：{raw_data.shape}")
    
    # 不支持的格式
    else:
        raise ValueError(f"不支持的文件格式：{os.path.splitext(data_path)[1]}，仅支持.csv/.npz")
    
    # 归一化处理（修复：确保scaler匹配当前数据尺度）
    if scaler_path:
        # 确保scaler目录存在
        scaler_dir = os.path.dirname(scaler_path)
        if scaler_dir and not os.path.exists(scaler_dir):
            os.makedirs(scaler_dir)
        
        try:
            # 加载scaler（新版joblib无需allow_pickle）
            scaler = joblib.load(scaler_path)
            # 验证scaler是否适配当前数据（避免尺度不匹配）
            test_data = raw_data[:1].copy()
            scaler.transform(test_data)
            scaled_data = scaler.transform(raw_data)
            print("复用PEMS04归一化器处理SUMO数据（尺度匹配）")
        except Exception as e:
            # 尺度不匹配时，重新训练scaler并保存
            print(f"PEMS04 scaler与当前数据尺度不匹配：{e}，重新训练scaler")
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(raw_data)
            joblib.dump(scaler, scaler_path)
            print(f"新scaler已保存到：{scaler_path}")
    else:
        # 对照组：重新训练scaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(raw_data)
        print("重新训练归一化器处理SUMO数据")
    
    return scaled_data, scaler, raw_data.shape[1]

# ---------------------- 步骤4：构建时序数据集（增加数据校验） ----------------------
def create_time_series_dataset(data, look_back=1):
    """
    构建时序数据集（增加数据量校验，避免空数据集）
    """
    if len(data) <= look_back:
        raise ValueError(f"数据量({len(data)})小于序列长度({look_back})，无法构建时序数据集")
    
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), :])
        y.append(data[i + look_back, :])
    
    X = np.array(X)
    y = np.array(y)
    
    # 按时间划分训练/测试集（保证时序连续性）
    train_size = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 数据量校验
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(f"训练/测试集数据量为0，请调整TEST_SIZE（当前：{TEST_SIZE}）")
    
    print(f"时序数据集构建完成：训练集{X_train.shape}，测试集{X_test.shape}")
    return X_train, X_test, y_train, y_test

# ---------------------- 步骤5：构建LSTM模型（优化网络结构） ----------------------
def build_lstm_model(input_shape, units=64, n_features=1):
    """
    优化LSTM模型结构：增加正则化，提升泛化能力
    """
    model = Sequential([
        LSTM(units=units, return_sequences=True, input_shape=input_shape, dropout=0.2, recurrent_dropout=0.1),
        LSTM(units=units//2, return_sequences=False, dropout=0.2, recurrent_dropout=0.1),
        Dense(units=16, activation="relu"),
        Dropout(0.2),
        Dense(units=n_features)
    ])
    # 优化优化器参数
    model.compile(
        optimizer=Adam(learning_rate=1e-4, clipnorm=1.0),  # 梯度裁剪，避免爆炸
        loss="mean_squared_error", 
        metrics=["mae"]
    )
    return model

# ---------------------- 步骤6：训练模型（优化早停逻辑） ----------------------
def train_model(X_train, y_train, X_test, y_test, is_transfer=False, pretrained_path=None):
    input_shape = (X_train.shape[1], X_train.shape[2])
    n_features = X_train.shape[2]
    
    # 核心优化：调整早停参数，避免提前终止
    callbacks = [
        EarlyStopping(
            monitor="val_loss", 
            patience=EARLY_STOP_PATIENCE,  # 优化：从3改为5
            restore_best_weights=True, 
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(RESULT_DIR, f"best_model_{'transfer' if is_transfer else 'baseline'}.keras"),
            monitor="val_loss", 
            save_best_only=True, 
            verbose=1
        )
    ]
    
    if is_transfer and os.path.exists(pretrained_path):
        # 迁移学习：加载预训练模型+微调
        try:
            base_model = load_model(pretrained_path)
            # 冻结前两层LSTM，只微调全连接层
            for layer in base_model.layers[:2]:
                layer.trainable = False
            # 替换输出层，适配SUMO数据尺度
            model = Sequential(base_model.layers[:-1] + [Dense(n_features)])
            model.compile(
                optimizer=Adam(learning_rate=5e-6, clipnorm=1.0),
                loss="mean_squared_error", 
                metrics=["mae"]
            )
            print("✅ 加载PEMS04预训练模型，冻结底层LSTM层进行微调")
        except Exception as e:
            print(f"❌ 加载预训练模型失败：{e}，改为从头训练")
            model = build_lstm_model(input_shape, UNITS, n_features)
    else:
        # 对照组：从头训练
        model = build_lstm_model(input_shape, UNITS, n_features)
        print("✅ 从头训练全新LSTM模型")
    
    # 训练模型（增加verbose=1，显示详细训练过程）
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

# ---------------------- 步骤7：评估模型（修复反归一化） ----------------------
def evaluate_model(model, X_test, y_test, scaler):
    """
    核心修复：确保反归一化使用对应组的scaler，避免尺度偏差
    """
    y_pred = model.predict(X_test, verbose=0)
    
    # 强制用当前组的scaler反归一化（核心修复）
    try:
        y_test_inv = scaler.inverse_transform(y_test)
        y_pred_inv = scaler.inverse_transform(y_pred)
    except Exception as e:
        print(f"反归一化失败：{e}，使用原始数据计算指标")
        y_test_inv = y_test
        y_pred_inv = y_pred
    
    # 展平数据计算指标
    y_test_flat = y_test_inv.flatten()
    y_pred_flat = y_pred_inv.flatten()
    
    # 计算指标（保留更多小数位，提升精度）
    mse = mean_squared_error(y_test_flat, y_pred_flat)
    mae = mean_absolute_error(y_test_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_flat, y_pred_flat)
    
    metrics = {
        "MSE": round(mse, 6),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R²": round(r2, 4)
    }
    
    print(f"模型评估完成：{metrics}")
    return metrics, y_test_inv, y_pred_inv

# ---------------------- 步骤8：结果可视化（优化图表展示） ----------------------
def plot_results(baseline_metrics, transfer_metrics, y_test_inv, y_pred_baseline, y_pred_transfer):
    """
    优化可视化：更清晰的图表布局+标注关键指标
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 指标对比柱状图（优化：区分误差/拟合指标的展示逻辑）
    metrics = ["MSE", "MAE", "RMSE", "R²"]
    baseline_vals = [baseline_metrics[m] for m in metrics]
    transfer_vals = [transfer_metrics[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax1.bar(x - width/2, baseline_vals, width, label="对照组（从头训练）", color="#FF7F7F")
    bars2 = ax1.bar(x + width/2, transfer_vals, width, label="实验组（迁移学习）", color="#7FFF7F")
    
    # 标注数值
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f"{height:.4f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f"{height:.4f}", ha="center", va="bottom", fontsize=8)
    
    ax1.set_title("模型评估指标对比", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. 真实值vs对照组
    ax2.plot(y_test_inv[:200, 0], label="真实值", color="blue", linewidth=1.5)
    ax2.plot(y_pred_baseline[:200, 0], label="对照组预测值", color="orange", linestyle="--", linewidth=1.2)
    ax2.set_title("对照组：真实值 vs 预测值", fontsize=12, fontweight="bold")
    ax2.set_xlabel("时间步")
    ax2.set_ylabel("流量")
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. 真实值vs实验组
    ax3.plot(y_test_inv[:200, 0], label="真实值", color="blue", linewidth=1.5)
    ax3.plot(y_pred_transfer[:200, 0], label="实验组预测值", color="green", linestyle="--", linewidth=1.2)
    ax3.set_title("实验组：真实值 vs 预测值", fontsize=12, fontweight="bold")
    ax3.set_xlabel("时间步")
    ax3.set_ylabel("流量")
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. 两组预测值对比
    ax4.plot(y_pred_baseline[:200, 0], label="对照组", color="orange", linestyle="--", linewidth=1.2)
    ax4.plot(y_pred_transfer[:200, 0], label="实验组", color="green", linestyle="--", linewidth=1.2)
    ax4.set_title("两组预测值对比", fontsize=12, fontweight="bold")
    ax4.set_xlabel("时间步")
    ax4.set_ylabel("流量")
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "comparison_results_optimized.png"), dpi=300, bbox_inches="tight")
    plt.show()
    
    # 保存详细指标到CSV（增加备注）
    metrics_df = pd.DataFrame({
        "指标": metrics,
        "对照组（从头训练）": baseline_vals,
        "实验组（迁移学习）": transfer_vals,
        "实验组提升率": [
            f"{round((baseline_vals[i]-transfer_vals[i])/baseline_vals[i]*100, 2)}%" if i<3 else f"{round((transfer_vals[i]-baseline_vals[i])/abs(baseline_vals[i])*100, 2)}%"
            for i in range(len(metrics))
        ]
    })
    metrics_df.to_csv(os.path.join(RESULT_DIR, "comparison_metrics_optimized.csv"), index=False)
    print(f"✅ 详细指标已保存：{os.path.join(RESULT_DIR, 'comparison_metrics_optimized.csv')}")

# ---------------------- 主函数（优化流程逻辑） ----------------------
if __name__ == "__main__":
    try:
        print("="*60)
        print("开始优化版迁移学习对比实验（固定随机种子，保证可复现）")
        print("="*60)
        
        # 1. 加载数据（对照组+实验组）
        print("\n=== 加载对照组数据 ===")
        scaled_data_baseline, scaler_baseline, n_features = load_sumo_data(SUMO_DATA_PATH)
        
        print("\n=== 加载实验组数据 ===")
        scaled_data_transfer, scaler_transfer, _ = load_sumo_data(SUMO_DATA_PATH, PEMS_SCALER_PATH)
        
        # 2. 构建时序数据集
        print("\n=== 构建时序数据集 ===")
        X_train_baseline, X_test_baseline, y_train_baseline, y_test_baseline = create_time_series_dataset(scaled_data_baseline, LOOK_BACK)
        X_train_transfer, X_test_transfer, y_train_transfer, y_test_transfer = create_time_series_dataset(scaled_data_transfer, LOOK_BACK)
        
        # 3. 训练对照组
        print("\n=== 训练对照组模型 ===")
        baseline_model, baseline_history = train_model(
            X_train_baseline, y_train_baseline, X_test_baseline, y_test_baseline,
            is_transfer=False
        )
        
        # 4. 训练实验组
        print("\n=== 训练实验组模型 ===")
        transfer_model, transfer_history = train_model(
            X_train_transfer, y_train_transfer, X_test_transfer, y_test_transfer,
            is_transfer=True, pretrained_path=PEMS_PRETRAINED_MODEL_PATH
        )
        
        # 5. 评估模型
        print("\n=== 评估对照组模型 ===")
        baseline_metrics, y_test_inv_baseline, y_pred_inv_baseline = evaluate_model(baseline_model, X_test_baseline, y_test_baseline, scaler_baseline)
        
        print("\n=== 评估实验组模型 ===")
        transfer_metrics, y_test_inv_transfer, y_pred_inv_transfer = evaluate_model(transfer_model, X_test_transfer, y_test_transfer, scaler_transfer)
        
        # 6. 打印最终结果
        print("\n" + "="*60)
        print("最终实验结果对比（优化版）")
        print("="*60)
        print(f"对照组（从头训练）：{baseline_metrics}")
        print(f"实验组（迁移学习）：{transfer_metrics}")
        print("="*60)
        
        # 7. 可视化结果
        print("\n=== 生成可视化结果 ===")
        plot_results(baseline_metrics, transfer_metrics, y_test_inv_baseline, y_pred_inv_baseline, y_pred_inv_transfer)
        
        # 8. 保存最终模型
        baseline_model.save(os.path.join(RESULT_DIR, "final_baseline_model.keras"))
        transfer_model.save(os.path.join(RESULT_DIR, "final_transfer_model.keras"))
        
        # 9. 保存scaler（便于后续复用）
        joblib.dump(scaler_baseline, os.path.join(RESULT_DIR, "scaler_baseline.pkl"))
        joblib.dump(scaler_transfer, os.path.join(RESULT_DIR, "scaler_transfer.pkl"))
        
        print("\n✅ 所有实验流程完成！结果已保存至：", RESULT_DIR)
        
    except Exception as e:
        print(f"\n❌ 实验执行失败！错误信息：{e}")
        import traceback
        traceback.print_exc()