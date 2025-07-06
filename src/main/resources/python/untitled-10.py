import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 模拟知乎或快手广告营销 CTR 数据集
# 假设每个样本有 20 个特征，最后一列为点击率标签
num_samples = 10000
num_features = 20
X = np.random.randn(num_samples, num_features).astype(np.float32)
y = np.random.randint(0, 2, num_samples).astype(np.float32)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 定义数据集类
class CTRDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

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
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout),
            num_layers=num_layers
        )
        self.moe = MoE(num_experts, d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(0)  # 添加序列维度
        x = self.transformer_encoder(x)
        x = x.squeeze(0)
        x = self.moe(x)
        return torch.sigmoid(x).squeeze()


# 初始化数据集和数据加载器
train_dataset = CTRDataset(X_train, y_train)
test_dataset = CTRDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 模型参数
INPUT_DIM = num_features
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
for epoch in range(N_EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion)
    test_loss = evaluate(model, test_loader, criterion)
    print(f'Epoch {epoch + 1}/{N_EPOCHS}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

print("Training finished.")
