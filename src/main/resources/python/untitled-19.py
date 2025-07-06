import os
import gzip
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据路径
DATA_DIR = "D:/code/data/llm/手撕LLM速成班-试听课-小冬瓜AIGC-20231211/data/avazu"
TRAIN_PATH = os.path.join(DATA_DIR, "train.gz")
TEST_PATH = os.path.join(DATA_DIR, "test.gz")

# 检查文件是否存在
if not all([os.path.exists(path) for path in [TRAIN_PATH, TEST_PATH]]):
    raise FileNotFoundError("请检查 Avazu 数据集文件是否存在于指定目录。")

# 读取并解压数据
def read_gz_file(file_path):
    with gzip.open(file_path, 'rt') as f:
        df = pd.read_csv(f)
    return df

train_df = read_gz_file(TRAIN_PATH)
test_df = read_gz_file(TEST_PATH)
print("读取并解压数据完成。")

# 数据预处理
# 训练集的 'click' 列是标签
categorical_columns = train_df.select_dtypes(include=['object']).columns
print(categorical_columns)

# 记录训练集的行数，用于后续拆分
train_rows = len(train_df)

# 合并训练集和测试集
combined_df = pd.concat([train_df, test_df], ignore_index=True)

for col in categorical_columns:
    le = LabelEncoder()
    combined_df[col] = le.fit_transform(combined_df[col].astype(str))

# 拆分回训练集和测试集
train_df = combined_df[:train_rows]
test_df = combined_df[train_rows:]

print("数据预处理完成。")

# 划分特征和标签
X = train_df.drop(['click'], axis=1).values
y = train_df['click'].values

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义数据集类
class AvazuDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).to(device)

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
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        self.moe = MoE(num_experts, d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.moe(x)
        return torch.sigmoid(x).squeeze()

# 初始化数据集和数据加载器
train_dataset = AvazuDataset(X_train, y_train)
val_dataset = AvazuDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
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
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# 训练循环
N_EPOCHS = 10
for epoch in range(N_EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion)
    val_loss = evaluate(model, val_loader, criterion)
    print(f'Epoch {epoch + 1}/{N_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

print("Training finished.")
