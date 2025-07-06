import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据路径，需要手动下载并解压数据集
DATA_PATH = 'round1_ijcai_18_train_20180301.txt'
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("请从阿里天池平台下载相关数据集并确保 round1_ijcai_18_train_20180301.txt 文件存在。")

# 读取数据
df = pd.read_csv(DATA_PATH, sep='\t')

# 数据预处理
# 选择部分特征，这里简单选择一些特征，实际可根据需求调整
selected_features = ['item_id', 'user_id', 'context_page_id', 'shop_id']
target = 'is_trade'

# 对类别特征进行编码
for col in selected_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

X = df[selected_features].values
y = df[target].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 定义数据集类
class AliCTRDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float)

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
    def __init__(self, num_features, num_classes, d_model=128, nhead=4, num_layers=2, num_experts=4, dropout=0.5):
        super(TransformerMoERecommender, self).__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(df[col].nunique(), d_model) for col in selected_features])
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model * num_features, nhead=nhead, dropout=dropout),
            num_layers=num_layers
        )
        self.moe = MoE(num_experts, d_model * num_features, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embeds = [self.embeddings[i](x[:, i]) for i in range(x.shape[1])]
        x = torch.cat(embeds, dim=1)
        x = x.unsqueeze(0)  # 添加序列维度
        x = self.transformer_encoder(x)
        x = x.squeeze(0)
        x = self.moe(x)
        return torch.sigmoid(x).squeeze()


# 初始化数据集和数据加载器
train_dataset = AliCTRDataset(X_train, y_train)
test_dataset = AliCTRDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 初始化模型、损失函数和优化器
num_features = len(selected_features)
num_classes = 1
model = TransformerMoERecommender(num_features, num_classes).to(device)
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
for epoch in range(N_EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion)
    test_loss = evaluate(model, test_loader, criterion)
    print(f'Epoch {epoch + 1}/{N_EPOCHS}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

print("Training finished.")
