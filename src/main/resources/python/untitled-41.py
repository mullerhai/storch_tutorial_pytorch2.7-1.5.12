import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import requests
import zipfile

# 下载并解压数据集
def download_and_extract(url, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    zip_path = os.path.join(save_path, 'creditcard.zip')
    if not os.path.exists(zip_path):
        print("Downloading dataset...")
        response = requests.get(url)
        with open(zip_path, 'wb') as f:
            f.write(response.content)
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(save_path)

# 自定义数据集类
class CreditFraudDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 专家网络
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 门控网络
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        x = self.softmax(x)
        return x

# 混合专家模型
class MoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim) for _ in range(num_experts)])
        self.gating_network = GatingNetwork(input_dim, num_experts)

    def forward(self, x):
        gates = self.gating_network(x)
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1).squeeze(-1)
        output = torch.sum(gates.unsqueeze(-1) * expert_outputs, dim=1)
        return output

# 评分卡模块
class ScoreCard(nn.Module):
    def __init__(self, input_dim):
        super(ScoreCard, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.fc(x)
        return x

# Transformer MoE 模型
class TransformerMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, nhead=4, num_layers=2):
        super(TransformerMoE, self).__init__()
        self.moe = MoE(input_dim, hidden_dim, num_experts)
        self.score_card = ScoreCard(input_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead),
            num_layers=num_layers
        )
        self.fc = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_transformer = self.transformer_encoder(x.unsqueeze(0)).squeeze(0)
        moe_output = self.moe(x_transformer)
        score_card_output = self.score_card(x_transformer)
        combined_output = moe_output + score_card_output
        output = self.fc(combined_output)
        output = self.sigmoid(output)
        return output

# 训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# 评估函数
def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(test_loader)

# 主函数
def main():
    # 下载并解压数据集
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/creditcard.zip'
    save_path = 'data'
    download_and_extract(url, save_path)

    # 读取数据
    data_path = os.path.join(save_path, 'creditcard.csv')
    df = pd.read_csv(data_path)

    # 数据预处理
    X = df.drop('Class', axis=1).values
    y = df['Class'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = CreditFraudDataset(X_train, y_train)
    test_dataset = CreditFraudDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    input_dim = X_train.shape[1]
    hidden_dim = 64
    num_experts = 4
    model = TransformerMoE(input_dim, hidden_dim, num_experts).to(device)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

if __name__ == "__main__":
    main()
