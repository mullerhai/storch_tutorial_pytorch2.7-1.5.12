import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 模拟淘宝电商数据集
class TaobaoDataset(Dataset):
    def __init__(self, num_users=1000, num_items=500, num_samples=10000):
        self.user_ids = torch.randint(0, num_users, (num_samples,))
        self.item_ids = torch.randint(0, num_items, (num_samples,))
        self.labels = torch.randint(0, 2, (num_samples,)).float()  # 0 表示未点击，1 表示点击

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.labels[idx]

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
    def __init__(self, num_users, num_items, d_model, nhead, num_layers, num_experts, dropout):
        super(TransformerMoERecommender, self).__init__()
        self.user_embedding = nn.Embedding(num_users, d_model)
        self.item_embedding = nn.Embedding(num_items, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout),
            num_layers=num_layers
        )
        self.moe = MoE(num_experts, d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, user_ids, item_ids):
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)
        combined_embed = user_embed + item_embed
        combined_embed = combined_embed.unsqueeze(0)  # 添加序列维度
        output = self.transformer_encoder(combined_embed)
        output = output.squeeze(0)
        output = self.moe(output)
        return torch.sigmoid(output).squeeze()

# 初始化数据集和数据加载器
train_dataset = TaobaoDataset()
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 模型参数
NUM_USERS = 1000
NUM_ITEMS = 500
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
NUM_EXPERTS = 4
DROPOUT = 0.5

# 初始化模型、损失函数和优化器
model = TransformerMoERecommender(NUM_USERS, NUM_ITEMS, D_MODEL, NHEAD, NUM_LAYERS, NUM_EXPERTS, DROPOUT).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for user_ids, item_ids, labels in dataloader:
        user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(user_ids, item_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# 训练循环
N_EPOCHS = 10
for epoch in range(N_EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion)
    print(f'Epoch {epoch + 1}/{N_EPOCHS}, Train Loss: {train_loss:.4f}')

print("Training finished.")
