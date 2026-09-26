# ---------------------- 步骤1：导入依赖库 ----------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 新增：配置Matplotlib支持中文显示，解决乱码问题
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "Heiti TC", "PingFang SC"]  # 适配Windows/Mac/Linux
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方块的问题

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # 新增r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import joblib  # 提前导入，避免函数内重复导入

# ---------------------- 步骤2：配置全局参数（可按需调整） ----------------------
# 数据路径（你的.npz文件路径，已确认键名为cleaned_traffic）
DATA_PATH = "cleaned_pems04.npz"
# 时序参数：输入序列长度（用前look_back个时间步预测下1个时间步）
LOOK_BACK = 12  # 对应5分钟间隔的1小时，数据量不足可改为6
# 模型训练参数
BATCH_SIZE = 64  # 数据量较小可改为32
EPOCHS = 50
UNITS = 64  # LSTM隐藏层神经元数量
TEST_SIZE = 0.2  # 测试集占比（按时间顺序划分，避免数据泄露）
# 保存路径（保存训练好的模型和归一化器）
SAVE_DIR = "trained_lstm_model"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# ---------------------- 步骤3：数据加载与预处理 ----------------------
def load_and_preprocess_data(data_path):
    """
    加载清洗后的数据，完成归一化和格式转换
    适配：一维.npz（cleaned_traffic）、.csv、.txt 格式
    修复：维度不匹配、键名错误、一维数组重塑问题
    """
    # 步骤3.1：先校验文件是否存在（提前报错，清晰易懂）
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"文件不存在！请检查路径是否正确：{data_path}")
    
    # 步骤3.2：加载数据（适配各类格式，重点处理一维.npz）
    target_data = None
    if data_path.endswith(".npz"):
        # 加载.npz格式（适配cleaned_traffic键，一维数据自动重塑）
        npz_data = np.load(data_path)
        target_key = "cleaned_traffic"
        
        # 校验键名是否存在
        if target_key not in npz_data.files:
            raise KeyError(f".npz文件中不存在键'{target_key}'，文件包含的键为：{npz_data.files}")
        
        # 加载一维数据
        raw_data = npz_data[target_key]
        print(f"成功加载.npz文件，原始数据形状：{raw_data.shape}")
        
        # 适配一维数组：重塑为二维数组（时间步 × 1个特征），满足后续流程要求
        if len(raw_data.shape) == 1:
            print("检测到一维数组，自动重塑为二维数组（时间步 × 1个特征）")
            target_data = raw_data.reshape(-1, 1)  # 重塑后shape: (6915, 1)
        elif len(raw_data.shape) >= 2:
            # 兼容后续可能的二维/三维数据
            target_data = raw_data[:, 0, :] if len(raw_data.shape) == 3 else raw_data
        else:
            raise ValueError(f"不支持的数组维度：{len(raw_data.shape)}")
    
    elif data_path.endswith(".csv"):
        # 加载.csv格式（支持流量/速度/占有率三特征）
        df = pd.read_csv(data_path)
        # 按时间戳排序（确保时序连续性）
        if "time" in df.columns:
            df = df.sort_values(by="time")
        # 选择核心特征，无多特征则取第一列
        if all(col in df.columns for col in ["flow", "speed", "occupancy"]):
            target_data = df[["flow", "speed", "occupancy"]].values
        else:
            target_data = df.iloc[:, 0].values.reshape(-1, 1)
        print(f"成功加载.csv文件，数据形状：{target_data.shape}")
    
    elif data_path.endswith(".txt"):
        # 加载.txt格式（无表头，空格/逗号分隔，自动适配一维/二维）
        try:
            raw_data = np.loadtxt(data_path, delimiter=" ")
        except:
            raw_data = np.loadtxt(data_path, delimiter=",")
        
        print(f"成功加载.txt文件，原始数据形状：{raw_data.shape}")
        # 重塑一维.txt数据为二维
        if len(raw_data.shape) == 1:
            target_data = raw_data.reshape(-1, 1)
        else:
            target_data = raw_data
    else:
        # 抛出清晰的格式错误提示
        file_ext = os.path.splitext(data_path)[1]
        raise ValueError(f"不支持的文件格式：{file_ext}！仅支持.npz、.csv和.txt格式。")
    
    # 步骤3.3：数据合法性校验（确保数值型，无无效数据）
    if not np.issubdtype(target_data.dtype, np.number):
        raise ValueError("数据不是数值型，无法进行归一化和模型训练")
    print(f"数据处理完成，最终形状：{target_data.shape}（时间步 × 特征数）")
    
    # 步骤3.4：数据归一化（LSTM对数据尺度敏感，必须归一化到[0,1]）
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(target_data)
    
    # 步骤3.5：保存归一化器（后续预测时需反归一化，还原真实数据尺度）
    scaler_save_path = os.path.join(SAVE_DIR, "minmax_scaler.pkl")
    joblib.dump(scaler, scaler_save_path)
    print(f"归一化器已保存至：{scaler_save_path}")
    
    # 返回：归一化数据、归一化器、特征数
    return scaled_data, scaler, target_data.shape[1]

# 执行数据加载与预处理（捕获异常，方便排查问题）
try:
    scaled_data, scaler, n_features = load_and_preprocess_data(DATA_PATH)
except Exception as e:
    print(f"数据加载失败！错误信息：{e}")
    exit()  # 加载失败则终止程序，避免后续报错

# ---------------------- 步骤4：构建时序数据集（滑动窗口） ----------------------
def create_time_series_dataset(data, look_back=1):
    """
    构建LSTM输入的时序数据集：
    输入X：(样本数, look_back, 特征数)（三维数组，LSTM要求的输入格式）
    输出y：(样本数, 特征数)（预测下一个时间步的特征）
    适配：单特征/多特征数据，按时间顺序划分训练集/测试集
    """
    X, y = [], []
    # 滑动窗口遍历数据，构建样本（避免索引越界）
    if len(data) <= look_back:
        raise ValueError(f"数据量({len(data)})小于序列长度({look_back})，无法构建时序数据集")
    
    for i in range(len(data) - look_back):
        # 提取输入序列：前look_back个时间步
        X.append(data[i:(i + look_back), :])
        # 提取输出标签：第i+look_back个时间步（下一个时间步，即预测目标）
        y.append(data[i + look_back, :])
    
    # 转换为numpy数组，适配Keras模型输入
    X = np.array(X)
    y = np.array(y)
    
    # 按时间顺序划分训练集和测试集（时序数据不可随机打乱，避免数据泄露）
    train_size = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test

# 执行时序数据集构建
try:
    X_train, X_test, y_train, y_test = create_time_series_dataset(scaled_data, LOOK_BACK)
except ValueError as e:
    print(f"数据集构建失败！错误信息：{e}")
    print("建议减小LOOK_BACK参数（如改为6），适配当前数据量")
    exit()

# 打印数据集形状，验证格式正确性
print("\n---------------------- 数据集形状验证 ----------------------")
print(f"训练集输入形状：X_train.shape = {X_train.shape}（样本数, 时间步, 特征数）")
print(f"训练集输出形状：y_train.shape = {y_train.shape}（样本数, 特征数）")
print(f"测试集输入形状：X_test.shape = {X_test.shape}")
print(f"测试集输出形状：y_test.shape = {y_test.shape}")

# ---------------------- 步骤5：定义LSTM模型 ----------------------
def build_lstm_model(input_shape, units=64, n_features=1):
    """
    定义基础LSTM模型（适配单特征/多特征输入输出）
    堆叠两层LSTM+Dropout，防止过拟合，提升模型泛化能力
    """
    model = Sequential([
        # 第一层LSTM：返回序列（堆叠多层LSTM时，前层必须返回序列）
        LSTM(units=units, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),  # 随机丢弃20%的神经元，防止过拟合
        # 第二层LSTM：不返回序列（最后一层LSTM无需返回序列）
        LSTM(units=units//2, return_sequences=False),
        Dropout(0.2),
        # 全连接层：输出与特征数一致，适配单特征/多特征预测
        Dense(units=n_features)
    ])
    
    # 编译模型：adam优化器（自适应学习率，收敛更快），MSE损失函数（回归任务首选）
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    
    return model

# 定义输入形状：(时间步长度, 特征数)
input_shape = (LOOK_BACK, n_features)
# 构建LSTM模型
lstm_model = build_lstm_model(input_shape, UNITS, n_features)
# 打印模型结构，验证层与形状是否正确
print("\n---------------------- LSTM模型结构 ----------------------")
lstm_model.summary()

# ---------------------- 步骤6：模型训练（加入回调函数防止过拟合） ----------------------
# 定义回调函数：监控训练过程，自动停止训练并保存最佳模型
callbacks = [
    # 早停回调：验证集损失连续5个epoch不下降则停止训练，避免过拟合
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    # 模型保存回调：监控验证集损失，只保存效果最好的模型
    ModelCheckpoint(
        os.path.join(SAVE_DIR, "best_lstm_model.keras"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

# 开始模型训练
print("\n---------------------- 开始训练LSTM模型 ----------------------")
history = lstm_model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=2
)

# ---------------------- 步骤7：模型评估与可视化（含MSE/MAE/R²完整指标） ----------------------
def evaluate_and_visualize(model, X_test, y_test, scaler, history):
    """
    用测试集评估模型性能，绘制损失曲线和预测结果对比图
    新增：MSE/MAE/R²完整评估指标，规范打印输出
    适配：单特征/多特征数据，中文图表无乱码
    """
    # 步骤7.1：用测试集进行预测
    y_pred = model.predict(X_test, verbose=0)
    
    # 步骤7.2：反归一化（将预测值和真实值还原为原始数据尺度，方便解读）
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)
    
    # 步骤7.3：计算完整评估指标（扁平化数据，适配单/多特征）
    y_test_flat = y_test_inv.flatten()
    y_pred_flat = y_pred_inv.flatten()
    mse = mean_squared_error(y_test_flat, y_pred_flat)
    mae = mean_absolute_error(y_test_flat, y_pred_flat)
    r2 = r2_score(y_test_flat, y_pred_flat)
    
    # 步骤7.4：打印规范评估结果
    print("\n---------------------- 模型评估结果 ----------------------")
    print(f"📊 模型评估结果：")
    print(f"  均方误差 (MSE): {mse:.4f}")
    print(f"  平均绝对误差 (MAE): {mae:.4f}")
    print(f"  决定系数 (R²): {r2:.4f}")
    print(f"  说明：MSE/MAE值越小越好，R²越接近1说明模型拟合效果越好")
    
    # 步骤7.5：提取第一特征用于可视化（对应流量/单一目标特征）
    feature_test = y_test_inv[:, 0]
    feature_pred = y_pred_inv[:, 0]
    
    # 步骤7.6：绘制训练/验证损失曲线（观察是否过拟合，中文无乱码）
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(history.history["loss"], label="训练损失", color="blue")
    plt.plot(history.history["val_loss"], label="验证损失", color="red")
    plt.title("LSTM模型训练/验证损失曲线")
    plt.xlabel("Epoch（训练轮次）")
    plt.ylabel("Loss（MSE，均方误差）")
    plt.legend()
    plt.grid(alpha=0.3, linestyle="--")
    
    # 步骤7.7：绘制预测结果对比图（取前200个时间步，便于观察趋势）
    plt.subplot(2, 1, 2)
    plt.plot(feature_test[:200], label="真实值", color="blue", linewidth=1.5)
    plt.plot(feature_pred[:200], label="预测值", color="orange", linestyle="--", linewidth=1.5)
    plt.title("LSTM模型预测结果对比（前200个时间步）")
    plt.xlabel("时间步")
    plt.ylabel("特征值（已反归一化，还原真实尺度）")
    plt.legend()
    plt.grid(alpha=0.3, linestyle="--")
    
    # 步骤7.8：保存可视化图表，方便后续查看
    fig_save_path = os.path.join(SAVE_DIR, "model_evaluation.png")
    plt.tight_layout()
    plt.savefig(fig_save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"\n评估图表已保存至：{fig_save_path}")

# 执行模型评估与可视化
evaluate_and_visualize(lstm_model, X_test, y_test, scaler, history)

# ---------------------- 步骤8：保存最终模型 ----------------------
final_model_save_path = os.path.join(SAVE_DIR, "final_lstm_model.keras")
lstm_model.save(final_model_save_path)
print(f"\n最终模型已保存至：{final_model_save_path}")
print(f"\n所有训练结果已保存至文件夹：{SAVE_DIR}")