import os
import gzip
import json
from operator import index

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 下载并加载 Amazon 数据集（以 Electronics 类别为例）
DATA_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz"
DATA_PATH = "reviews_Electronics_5.json.gz"
print("Downloading and loading Amazon dataset...")
if not os.path.exists(DATA_PATH):
    import urllib.request
    print("Downloading Amazon dataset...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)

# 解析数据
user_item_dict = defaultdict(list)
item_user_dict = defaultdict(list)
user_id_map = {}
item_id_map = {}
user_counter = 0
item_counter = 0
print("Loading Amazon dataset ungzip ...")
with gzip.open(DATA_PATH, 'rt', encoding='utf - 8') as f:
    for line in f:
        data = json.loads(line)
        user_id = data['reviewerID']
        item_id = data['asin']

        if user_id not in user_id_map:
            user_id_map[user_id] = user_counter
            user_counter += 1
        if item_id not in item_id_map:
            item_id_map[item_id] = item_counter
            item_counter += 1

        user_idx = user_id_map[user_id]
        item_idx = item_id_map[item_id]
        user_item_dict[user_idx].append(item_idx)
        item_user_dict[item_idx].append(user_idx)

# 生成训练和测试数据
all_interactions = []
for user in user_item_dict:
    for item in user_item_dict[user]:
        all_interactions.append((user, item, 1))  # 1 表示有交互

print("Generating training and testing data...")
# 生成负样本
num_negatives = 4
index =0
for user in user_item_dict:
    index += 1
    all_items = set(range(item_counter))
    interacted_items = set(user_item_dict[user])
    non_interacted_items = all_items - interacted_items
    non_interacted_items = list(non_interacted_items)
    sampled_negatives = np.random.choice(non_interacted_items, num_negatives * len(interacted_items))
    print("user_id: ", user, "index: ", index)
    for item in sampled_negatives:
        all_interactions.append((user, item, 0))  # 0 表示无交互
print("Training and testing data generated.")
train_interactions, test_interactions = train_test_split(all_interactions, test_size=0.2, random_state=42)




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


# 定义数据集类
class AmazonDataset(Dataset):
    def __init__(self, interactions):
        self.interactions = interactions

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        user, item, label = self.interactions[idx]
        return torch.tensor(user, dtype=torch.long), torch.tensor(item, dtype=torch.long), torch.tensor(label,
                                                                                                       dtype=torch.float)

print("Initializing Amazon dataset and data loaders...")
# 初始化数据集和数据加载器
train_dataset = AmazonDataset(train_interactions)
test_dataset = AmazonDataset(test_interactions)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print("Amazon dataset and data loaders initialized complete.")
# 模型参数
NUM_USERS = user_counter
NUM_ITEMS = item_counter
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
    print("Starting training...")
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
        print(f"Batch loss: {loss.item()}")
    return total_loss / len(dataloader)


# 评估函数
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    print("Starting evaluation...")
    with torch.no_grad():
        for user_ids, item_ids, labels in dataloader:
            user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
            outputs = model(user_ids, item_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            print(f"evaluation Batch loss: {loss.item()}")
    return total_loss / len(dataloader)


# 训练循环
N_EPOCHS = 10
for epoch in range(N_EPOCHS):
    print(f"EpochRange {epoch + 1}/{N_EPOCHS}")
    train_loss = train(model, train_loader, optimizer, criterion)
    test_loss = evaluate(model, test_loader, criterion)
    print(f'Epoch {epoch + 1}/{N_EPOCHS}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

print("Training finished.")
