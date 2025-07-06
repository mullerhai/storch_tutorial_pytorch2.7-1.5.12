import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载数据
def load_data():
    train_transaction = pd.read_csv('train_transaction.csv')
    train_identity = pd.read_csv('train_identity.csv')
    train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
    return train

# 数据预处理
def preprocess_data(data):
    # 处理缺失值
    data = data.fillna(-999)
    # 提取特征和标签
    X = data.drop(['isFraud', 'TransactionID'], axis=1)
    y = data['isFraud']
    # 处理类别特征
    cat_cols = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=cat_cols)
    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

# 自定义数据集类
class CreditFraudDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 定义专家网络
class Expert(nn.Module):
    def __init__(self, input_dim, d_model):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, d_model)
        self.fc2 = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义门控网络
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=1)

# 定义 MoE 模块
class MoE(nn.Module):
    def __init__(self, input_dim, d_model, num_experts):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList([Expert(input_dim, d_model) for _ in range(num_experts)])
        self.gating_network = GatingNetwork(input_dim, num_experts)

    def forward(self, x):
        gates = self.gating_network(x)
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        moe_output = torch.sum(gates.unsqueeze(-1) * expert_outputs, dim=1)
        return moe_output

# 定义 Transformer 块
class TransformerBlock(nn.Module):
    def __init__(self, input_dim, d_model, num_experts, d_ff, nhead, dropout):
        super(TransformerBlock, self).__init__()
        self.moe = MoE(input_dim, d_model, num_experts)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x):
        moe_output = self.moe(x)
        x = self.norm1(x + self.dropout1(moe_output))

        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm2(x + self.dropout2(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        return x

# 定义完整模型
class CreditFraudModel(nn.Module):
    def __init__(self, input_dim, d_model, num_experts, d_ff, nhead, num_layers, num_classes, dropout):
        super(CreditFraudModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, d_model, num_experts, d_ff, nhead, dropout) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.fc(x)
        return x

# 构建评分卡
def build_scorecard(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        outputs = model(X_tensor)
        scores = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
    min_score = scores.min()
    max_score = scores.max()
    scorecard = (scores - min_score) / (max_score - min_score) * 100
    return scorecard

# 主函数
def main():
    # 加载和预处理数据
    data = load_data()
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建数据集和数据加载器
    train_dataset = CreditFraudDataset(X_train, y_train)
    test_dataset = CreditFraudDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 模型参数
    input_dim = X_train.shape[1]
    d_model = 128
    num_experts = 4
    d_ff = 512
    nhead = 4
    num_layers = 2
    num_classes = 2
    dropout = 0.1

    # 初始化模型、损失函数和优化器
    model = CreditFraudModel(input_dim, d_model, num_experts, d_ff, nhead, num_layers, num_classes, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

    # 评估模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

    # 构建评分卡
    scorecard = build_scorecard(model, X_test)
    print("Sample scorecard values:", scorecard[:10])

if __name__ == "__main__":
    main()
