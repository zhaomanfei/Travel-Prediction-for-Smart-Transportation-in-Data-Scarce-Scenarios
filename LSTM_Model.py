import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib

# ---------------------- 第一步：加载预处理后的数据 ----------------------
def load_preprocessed_data(data_path):
    x_train = np.load(os.path.join(data_path, "x_train.npy"))
    y_train = np.load(os.path.join(data_path, "y_train.npy"))
    x_test = np.load(os.path.join(data_path, "x_test.npy"))
    y_test = np.load(os.path.join(data_path, "y_test.npy"))
    scaler = joblib.load(os.path.join(data_path, "scaler.pkl"))
    
    print(f"✅ 预处理数据加载成功：")
    print(f"  x_train.shape: {x_train.shape} (样本数, 输入序列长度, 传感器数)")
    print(f"  y_train.shape: {y_train.shape} (样本数, 预测序列长度, 传感器数)")
    print(f"  x_test.shape: {x_test.shape}")
    print(f"  y_test.shape: {y_test.shape}")
    return x_train, y_train, x_test, y_test, scaler

# ---------------------- 第二步：构建双层LSTM模型（修复输出形状不匹配）----------------------
def build_double_lstm_model(input_shape, output_seq_len, num_sensors):
    """
    修复关键：LSTM层2设return_sequences=False，用TimeDistributed包装全连接层
    或：LSTM层2输出后加一个“时间步维度压缩”，确保输出序列长度=output_seq_len
    """
    model = Sequential(name="Double_LSTM_Traffic_Prediction")
    
    # LSTM层1：64个神经元，返回完整序列（供下一层LSTM）
    model.add(LSTM(
        units=64,
        return_sequences=True,  # 第一层必须True（传递给第二层LSTM）
        input_shape=input_shape,
        dropout=0.2,
        recurrent_dropout=0.1,
        name="lstm_layer_1"
    ))
    
    # LSTM层2：32个神经元，不返回完整序列（只输出最后一个时间步的隐藏状态）
    model.add(LSTM(
        units=32,
        return_sequences=False,  # 关键修改：False→只输出最后1个时间步
        dropout=0.2,
        recurrent_dropout=0.1,
        name="lstm_layer_2"
    ))
    
    # 新增：全连接层+RepeatVector，将输出扩展为output_seq_len个时间步（适配3个预测步）
    model.add(Dense(units=32, activation="relu"))  # 中间全连接层增强拟合能力
    model.add(tf.keras.layers.RepeatVector(output_seq_len))  # 复制为3个时间步：(None, 3, 32)
    
    # 全连接层（用TimeDistributed包装，对每个时间步独立预测）
    model.add(tf.keras.layers.TimeDistributed(
        Dense(units=num_sensors, activation="linear"),  # 每个时间步输出862个传感器值
        name="output_layer"
    ))
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )
    
    model.summary()
    return model

# ---------------------- 第三步：模型训练与验证 ----------------------
def train_model(model, x_train, y_train, x_test, y_test, epochs=50, batch_size=32):
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    
    model_save_path = "D:/highSchool/科研/paper/LSTM/Pro/LSTMTrafficPro/best_model/"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(model_save_path, "double_lstm_best.h5"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping, checkpoint],
        shuffle=False
    )
    
    return history, model

# ---------------------- 第四步：模型评估与可视化 ----------------------
def evaluate_model(model, x_test, y_test, scaler, num_sensors=1, history=None):
    y_pred = model.predict(x_test, verbose=1)
    
    # 反归一化
    y_test_reshaped = y_test.reshape(-1, num_sensors)
    y_pred_reshaped = y_pred.reshape(-1, num_sensors)
    
    y_test_inv = scaler.inverse_transform(y_test_reshaped)
    y_pred_inv = scaler.inverse_transform(y_pred_reshaped)
    
    y_test_inv = y_test_inv.reshape(y_test.shape)
    y_pred_inv = y_pred_inv.reshape(y_pred.shape)
    
    # 计算指标
    mse = mean_squared_error(y_test_inv.flatten(), y_pred_inv.flatten())
    mae = mean_absolute_error(y_test_inv.flatten(), y_pred_inv.flatten())
    r2 = r2_score(y_test_inv.flatten(), y_pred_inv.flatten())
    
    print(f"\n📊 模型评估结果：")
    print(f"  均方误差 (MSE): {mse:.4f}")
    print(f"  平均绝对误差 (MAE): {mae:.4f}")
    print(f"  决定系数 (R²): {r2:.4f}")
    
    # 可视化预测结果
    sensor_idx = 0 
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号   
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_inv[:100, :, sensor_idx].flatten(), label="真实值", color="blue")
    plt.plot(y_pred_inv[:100, :, sensor_idx].flatten(), label="预测值", color="red", alpha=0.7)
    plt.xlabel("时间步（5分钟/步）")
    plt.ylabel("交通流量")
    plt.title(f"双层LSTM模型预测结果（传感器{sensor_idx+1}）")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("D:/highSchool/科研/paper/LSTM/Pro/LSTMTrafficPro/prediction_result.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # 可视化训练历史
    if history is not None:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="训练损失")
        plt.plot(history.history["val_loss"], label="验证损失")
        plt.xlabel("Epoch")
        plt.ylabel("损失值（MSE）")
        plt.title("训练/验证损失曲线")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history["mae"], label="训练MAE")
        plt.plot(history.history["val_mae"], label="验证MAE")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.title("训练/验证MAE曲线")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("D:/highSchool/科研/paper/LSTM/Pro/LSTMTrafficPro/training_history.png", dpi=300, bbox_inches="tight")
        plt.show()
    
    return mse, mae, r2, y_pred_inv

# ---------------------- 主程序 ----------------------
if __name__ == "__main__":
    preprocessed_data_path = "D:/highSchool/科研/paper/LSTM/Pro/LSTMTrafficPro/preprocessed_data/"
    x_train, y_train, x_test, y_test, scaler = load_preprocessed_data(preprocessed_data_path)
    
    input_shape = (x_train.shape[1], x_train.shape[2])  # (12, 862)
    output_seq_len = y_train.shape[1]  # 3（预测3个时间步）
    num_sensors = x_train.shape[2]  # 862（传感器数）
    
    model = build_double_lstm_model(input_shape, output_seq_len, num_sensors)
    
    history, trained_model = train_model(model, x_train, y_train, x_test, y_test, epochs=50, batch_size=32)
    
    mse, mae, r2, y_pred_inv = evaluate_model(trained_model, x_test, y_test, scaler, num_sensors, history)