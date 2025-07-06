import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据路径
DATA_DIR = "D:/code/data/llm/手撕LLM速成班-试听课-小冬瓜AIGC-20231211/data/criteo_small"
TRAIN_PATH = os.path.join(DATA_DIR, "train.txt")
TEST_PATH = os.path.join(DATA_DIR, "test.txt")
VALS_PATH = os.path.join(DATA_DIR, "val.txt")

# 检查文件是否存在
if not all([os.path.exists(path) for path in [TRAIN_PATH, TEST_PATH, VALS_PATH]]):
    raise FileNotFoundError("请检查数据文件是否存在于指定目录。")

# 读取数据
def read_data(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None)
    return df

train_df = read_data(TRAIN_PATH)
test_df = read_data(TEST_PATH)
vals_df = read_data(VALS_PATH)

# 数据预处理
# 第 0 列是标签，1 - 13 列是数值特征，14 - 39 列是类别特征
label_col = 0
numerical_cols = list(range(1, 14))
categorical_cols = list(range(14, 40))

# 合并数据集用于统一编码
all_df = pd.concat([train_df, test_df, vals_df], ignore_index=True)

# 处理缺失值
all_df[categorical_cols] = all_df[categorical_cols].fillna('nan')
all_df[numerical_cols] = all_df[numerical_cols].fillna(0)

# 对类别特征进行编码
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    all_df[col] = le.fit_transform(all_df[col].astype(str))
    label_encoders[col] = le

# 拆分回训练集、测试集和验证集
train_df = all_df[:len(train_df)]
test_df = all_df[len(train_df):len(train_df) + len(test_df)]
vals_df = all_df[len(train_df) + len(test_df):]

# 划分特征和标签
def split_features_labels(df):
    X = df.drop(columns=[label_col]).values
    y = df[label_col].values
    return X, y

X_train, y_train = split_features_labels(train_df)
X_test, y_test = split_features_labels(test_df)
X_val, y_val = split_features_labels(vals_df)

# 定义数据集类
class CriteoDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 定义 MoE 层
class MoE(nn.Module):
    def __init__(self, num_experts, d_model, num_classes):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList([nn.Linear(d_model, num_classes) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        gate_output = self.gate(x)
        gate_weights = torch.softmax(gate_output, dim=1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        output = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=1)
        return output

# 定义 Transformer MoE 推荐模型
class TransformerMoERecommender(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, num_experts, num_classes, dropout):
        super(TransformerMoERecommender, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout),
            num_layers=num_layers
        )
        self.moe = MoE(num_experts, d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = x.unsqueeze(1)  # 添加序列维度
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        x = self.moe(x)
        return torch.sigmoid(x).squeeze()

# 定义位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(0)]
        return x

# 初始化数据集和数据加载器
train_dataset = CriteoDataset(X_train, y_train)
test_dataset = CriteoDataset(X_test, y_test)
val_dataset = CriteoDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 模型参数
INPUT_DIM = X_train.shape[1]
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
NUM_EXPERTS = 4
NUM_CLASSES = 1
DROPOUT = 0.5

# 初始化模型、损失函数和优化器
model = TransformerMoERecommender(INPUT_DIM, D_MODEL, NHEAD, NUM_LAYERS, NUM_EXPERTS, NUM_CLASSES, DROPOUT).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# 评估函数
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# 训练循环
N_EPOCHS = 10
best_val_loss = float('inf')
for epoch in range(N_EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion)
    val_loss = evaluate(model, val_loader, criterion)
    print(f'Epoch {epoch + 1}/{N_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')

test_loss = evaluate(model, test_loader, criterion)
print(f'Test Loss: {test_loss:.4f}')
print("Training finished.")
