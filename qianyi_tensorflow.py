import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# ===================== 前置配置（替换为你的实际参数） =====================
# 1. 文件路径
KERAS_MODEL_PATH = "final_lstm_model.keras"
SUMO_TRAFFIC_CSV = "traffic_road_level_big.csv"

# 2. PEMS数据关键参数
pems_mean = 89.2  # 替换为你的PEMS流量均值
pems_std = 67.5   # 替换为你的PEMS流量标准差
seq_len = 12      # Keras模型输入历史时间步
pred_len = 3      # Keras模型输出预测时间步
n_features = 1    # 输入特征数（仅流量）
hidden_dim = 128  # Keras模型LSTM隐藏层维度
num_layers = 2    # Keras模型LSTM层数

# 3. 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")

# ===================== 步骤1：读取Keras模型，提取核心结构参数 =====================
def get_keras_model_params(keras_model_path):
    """加载Keras模型，提取结构参数"""
    keras_model = load_model(keras_model_path)
    print("成功加载Keras模型：final_lstm_model.keras")
    keras_model.summary()
    
    # 提取PEMS路段数
    input_shape = keras_model.input_shape
    n_nodes_pems = input_shape[2]
    
    return n_nodes_pems, keras_model

n_nodes_pems, keras_model = get_keras_model_params(KERAS_MODEL_PATH)

# ===================== 步骤2：重新定义PyTorch LSTM结构（修正输入维度逻辑） =====================
class TrafficLSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, n_nodes, n_features, pred_len):
        super(TrafficLSTM, self).__init__()
        self.input_size = input_size  # 修正：明确输入维度（n_nodes * n_features）
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_nodes = n_nodes
        self.n_features = n_features
        self.pred_len = pred_len
        
        # 底层LSTM（修正：input_size与输入数据最后一维一致）
        self.lstm_layers = nn.LSTM(
            input_size=self.input_size,  # 关键修正：不再动态计算，直接传入正确值（15）
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        
        # 顶层全连接层（适配SUMO路段数）
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_nodes * pred_len * n_features)
        )
    
    def forward(self, x):
        """
        前向传播（修正输入数据处理逻辑）
        :param x: 输入张量，形状：(batch, seq_len, n_nodes, n_features)
        :return: 输出张量，形状：(batch, pred_len, n_nodes, n_features)
        """
        batch_size = x.shape[0]
        
        # 关键修正：展平路段和特征维度，确保最后一维为input_size（15）
        # 转换后形状：(batch, seq_len, n_nodes * n_features) → (batch, 12, 15)
        x = x.reshape(batch_size, x.shape[1], -1)
        
        # LSTM层前向传播（此时x.size(-1)=15，与input_size=15一致）
        lstm_out, (_, _) = self.lstm_layers(x)  # lstm_out形状：(batch, seq_len, hidden_dim)
        lstm_out = lstm_out[:, -1, :]  # 取最后一个时间步输出
        
        # 全连接层前向传播
        fc_out = self.fc_layers(lstm_out)
        
        # 重塑输出形状
        output = fc_out.reshape(batch_size, self.pred_len, self.n_nodes, self.n_features)
        
        return output

# ===================== 步骤3：SUMO数据格式转换（保持不变，确保数据正确） =====================
def process_sumo_data(sumo_csv_path, pems_mean, pems_std, seq_len, pred_len, n_features):
    sumo_df = pd.read_csv(sumo_csv_path)
    sumo_df = sumo_df[["time", "edge_id", "flow"]].dropna()
    sumo_df = sumo_df.sort_values(by=["time", "edge_id"]).reset_index(drop=True)
    
    unique_times = sorted(sumo_df["time"].unique())
    unique_edges = sorted(sumo_df["edge_id"].unique())
    n_nodes_sumo = len(unique_edges)
    
    flow_matrix = np.zeros((len(unique_times), n_nodes_sumo))
    for t_idx, t in enumerate(unique_times):
        t_data = sumo_df[sumo_df["time"] == t]
        for e_idx, edge in enumerate(unique_edges):
            flow_val = t_data[t_data["edge_id"] == edge]["flow"].values
            flow_matrix[t_idx, e_idx] = flow_val[0] if len(flow_val) > 0 else 0
    
    flow_matrix_norm = (flow_matrix - pems_mean) / pems_std
    flow_matrix_norm = np.clip(flow_matrix_norm, -5, 5)
    
    def create_sequences(matrix, seq_len, pred_len):
        X, y = [], []
        for i in range(len(matrix) - seq_len - pred_len + 1):
            X.append(matrix[i:i+seq_len])
            y.append(matrix[i+seq_len:i+seq_len+pred_len])
        return np.array(X), np.array(y)
    
    X_sumo, y_sumo = create_sequences(flow_matrix_norm, seq_len, pred_len)
    
    X_sumo = X_sumo.reshape(X_sumo.shape[0], X_sumo.shape[1], X_sumo.shape[2], n_features)
    y_sumo = y_sumo.reshape(y_sumo.shape[0], y_sumo.shape[1], y_sumo.shape[2], n_features)
    
    X_sumo = torch.tensor(X_sumo, dtype=torch.float32).to(device)
    y_sumo = torch.tensor(y_sumo, dtype=torch.float32).to(device)
    
    train_split = int(0.8 * len(X_sumo))
    X_train, X_val = X_sumo[:train_split], X_sumo[train_split:]
    y_train, y_val = y_sumo[:train_split], y_sumo[train_split:]
    
    print(f"SUMO数据处理完成！")
    print(f"SUMO路段数：{n_nodes_sumo} | 输入形状：{X_train.shape}")
    print(f"训练集样本数：{len(X_train)} | 验证集样本数：{len(X_val)}")
    
    return X_train, X_val, y_train, y_val, n_nodes_sumo

# 处理SUMO数据
X_sumo_train, X_sumo_val, y_sumo_train, y_sumo_val, n_nodes_sumo = process_sumo_data(
    SUMO_TRAFFIC_CSV,
    pems_mean,
    pems_std,
    seq_len,
    pred_len,
    n_features
)

# ===================== 步骤4：初始化模型（关键修正：传入正确的input_size） =====================
# 计算正确的input_size：SUMO路段数 × 特征数 = 15 × 1 = 15
input_size_sumo = n_nodes_sumo * n_features

# 初始化PyTorch模型（传入正确的input_size_sumo=15）
model = TrafficLSTM(
    input_size=input_size_sumo,  # 关键修正：不再使用PEMS路段数，改用SUMO的输入维度15
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    n_nodes=n_nodes_sumo,  # 适配SUMO的15个路段
    n_features=n_features,
    pred_len=pred_len
).to(device)

# 迁移学习配置：冻结底层LSTM，微调顶层全连接
for param in model.lstm_layers.parameters():
    param.requires_grad = False
print("已冻结底层LSTM层，仅微调顶层全连接层")

# 重新确认顶层全连接层（适配SUMO路段数，无需修改）
model.fc_layers = nn.Sequential(
    nn.Linear(model.hidden_dim, 256),
    nn.ReLU(),
    nn.Linear(256, n_nodes_sumo * pred_len * n_features)
).to(device)

# 配置优化器和损失函数
criterion = nn.MSELoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-5
)

# ===================== 步骤5：模型微调训练与评估（保持不变） =====================
def train_model(model, X_train, y_train, X_val, y_val, criterion, optimizer, epochs=20, batch_size=32):
    train_loss_history = []
    val_loss_history = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
        
        avg_train_loss = train_loss / len(X_train)
        train_loss_history.append(avg_train_loss)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                batch_X = X_val[i:i+batch_size]
                batch_y = y_val[i:i+batch_size]
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_X.size(0)
        
        avg_val_loss = val_loss / len(X_val)
        val_loss_history.append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
    
    torch.save(model.state_dict(), "transfer_sumo_pytorch_model.pth")
    print("迁移学习模型保存完成：transfer_sumo_pytorch_model.pth")
    
    return model, train_loss_history, val_loss_history

# 微调训练模型
trained_model, train_loss, val_loss = train_model(
    model,
    X_sumo_train,
    y_sumo_train,
    X_sumo_val,
    y_sumo_val,
    criterion,
    optimizer,
    epochs=20,
    batch_size=32
)

# 模型评估
def evaluate_model(model, X_val, y_val, pems_mean, pems_std, batch_size=32):
    model.eval()
    y_pred_list = []
    y_true_list = []
    
    with torch.no_grad():
        for i in range(0, len(X_val), batch_size):
            batch_X = X_val[i:i+batch_size]
            batch_y = y_val[i:i+batch_size]
            
            outputs = model(batch_X)
            
            y_pred_list.append(outputs.cpu().numpy())
            y_true_list.append(batch_y.cpu().numpy())
    
    y_pred = np.concatenate(y_pred_list, axis=0)
    y_true = np.concatenate(y_true_list, axis=0)
    
    y_pred_denorm = y_pred * pems_std + pems_mean
    y_true_denorm = y_true * pems_std + pems_mean
    
    mae = np.mean(np.abs(y_true_denorm - y_pred_denorm))
    rmse = np.sqrt(np.mean((y_true_denorm - y_pred_denorm) ** 2))
    
    print("\n" + "="*50)
    print("迁移学习模型评估结果（反归一化后）")
    print("="*50)
    print(f"MAE（平均绝对误差）：{mae:.2f}")
    print(f"RMSE（均方根误差）：{rmse:.2f}")
    
    return mae, rmse

# 评估模型
mae_sumo, rmse_sumo = evaluate_model(
    trained_model,
    X_sumo_val,
    y_sumo_val,
    pems_mean,
    pems_std
)