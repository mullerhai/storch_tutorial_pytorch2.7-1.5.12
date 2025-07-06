import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import gzip
from sklearn.preprocessing import StandardScaler

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据路径配置（需手动从Criteo官网下载https://ailab.criteo.com/downloads/）
DATA_PATH = 'criteo_2022_sampled.txt.gz'
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("请从Criteo官网下载2022简化版数据集并确保文件存在")

# 读取数据（示例读取100万行，实际可调整）
num_rows = 1000000
with gzip.open(DATA_PATH, 'rt') as f:
    df = pd.read_csv(f, sep='\t', nrows=num_rows, 
                    names=['label'] + [f'num_feat_{i}' for i in range(13)] + [f'cat_feat_{i}' for i in range(26)])

# 数据预处理优化版
## 分离数值特征和类别特征
num_feats = [col for col in df.columns if 'num' in col]
cat_feats = [col for col in df.columns if 'cat' in col]
label_col = 'label'

## 数值特征处理（均值填充+标准化）
scaler = StandardScaler()
df[num_feats] = df[num_feats].fillna(df[num_feats].mean())
df[num_feats] = scaler.fit_transform(df[num_feats])

## 类别特征处理（哈希编码，处理OOV问题）
def hash_encode(series, hash_dim=10000):
    return series.apply(lambda x: hash(x) % hash_dim)

for col in cat_feats:
    df[col] = hash_encode(df[col].fillna('missing'), hash_dim=10000)  # 缺失值标记为'missing'

# 划分特征和标签
X = df[num_feats + cat_feats].values
y = df[label_col].values.astype(np.float32)

# 划分训练集和测试集（分层抽样保持正负样本比例）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# 定义优化后的数据集类（增加pin_memory提升GPU加载效率）
class Criteo2022Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 保持MoE层定义（与原代码一致）
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


# 调整后的Transformer MoE模型（适配新特征维度）
class TransformerMoERecommender(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, num_experts, num_classes, dropout):
        super(TransformerMoERecommender, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )  # 增加非线性激活和Dropout
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True),  # 启用batch_first
            num_layers=num_layers
        )
        self.moe = MoE(num_experts, d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # [B, D] -> [B, 1, D]
        x = self.transformer_encoder(x)  # [B, 1, D]
        x = x.squeeze(1)  # [B, D]
        return torch.sigmoid(self.moe(x)).squeeze()


# 初始化数据集和数据加载器（优化数据加载参数）
train_dataset = Criteo2022Dataset(X_train, y_train)
test_dataset = Criteo2022Dataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset, 
    batch_size=1024,  # 增大batch_size利用GPU并行
    shuffle=True,
    pin_memory=True,  # 锁页内存加速GPU传输
    num_workers=4  # 多进程加载
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=2048,
    shuffle=False,
    pin_memory=True,
    num_workers=4
)

# 模型参数（根据新数据集特征数量调整）
INPUT_DIM = len(num_feats) + len(cat_feats)  # 13+26=39维特征
D_MODEL = 256  # 增大模型容量
NHEAD = 8  # 更多注意力头
NUM_LAYERS = 3  # 增加Transformer层数
NUM_EXPERTS = 6  # 更多专家网络
NUM_CLASSES = 1
DROPOUT = 0.3  # 调整Dropout率

# 初始化模型（使用更高效的优化器）
model = TransformerMoERecommender(INPUT_DIM, D_MODEL, NHEAD, NUM_LAYERS, NUM_EXPERTS, NUM_CLASSES, DROPOUT).to(device)
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)  # 使用AdamW+权重衰减


# 训练函数（增加梯度裁剪）
def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪防止爆炸
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


# 评估函数（增加指标计算）
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


# 训练循环（增加早停机制）
N_EPOCHS = 20
best_test_loss = float('inf')
patience = 3
early_stop_counter = 0

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion)
    test_loss = evaluate(model, test_loader, criterion)
    
    print(f'Epoch {epoch+1}/{N_EPOCHS}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
    
    # 早停机制
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), 'best_ctr_model.pth')  # 保存最佳模型
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered")
            break

print("Training finished. Best test loss: {:.4f}".format(best_test_loss))
