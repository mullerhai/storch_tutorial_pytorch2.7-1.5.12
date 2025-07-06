import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 模拟用户、航班和景点数据
num_users = 1000
num_flights = 500
num_attractions = 300

# 模拟用户与航班、景点的交互数据
user_flight_interactions = np.random.randint(0, 2, size=(num_users, num_flights))
user_attraction_interactions = np.random.randint(0, 2, size=(num_users, num_attractions))

# 生成训练数据
all_data = []
print("Generating training data...")
for user_id in range(num_users):
    print(f"Generating data for user {user_id}...")
    for flight_id in range(num_flights):
        all_data.append((user_id, flight_id, user_flight_interactions[user_id][flight_id], 0))  # 0 表示航班
    for attraction_id in range(num_attractions):
        all_data.append((user_id, attraction_id + num_flights, user_attraction_interactions[user_id][attraction_id], 1))  # 1 表示景点
print("Training data generated.")
# 划分训练集和测试集
train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)

# 定义数据集类
class FlightAttractionDataset(Dataset):
    def __init__(self, data):
        self.data = data
        print("Dataset initialized.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_id, item_id, label, item_type = self.data[idx]
        return torch.tensor(user_id, dtype=torch.long), torch.tensor(item_id, dtype=torch.long), torch.tensor(label, dtype=torch.float), torch.tensor(item_type, dtype=torch.long)

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
print("Initializing dataset and dataloader...")
train_dataset = FlightAttractionDataset(train_data)
test_dataset = FlightAttractionDataset(test_data)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print("Dataset and dataloader initialized. completed")
# 模型参数
NUM_USERS = num_users
NUM_ITEMS = num_flights + num_attractions
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
    print("Starting training...")
    userIndex =0
    for user_ids, item_ids, labels, _ in dataloader:
        # print(f"Training on batch with shape {user_ids.shape}")
        user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
        optimizer.zero_grad()
        # print("model Forward pass...")
        outputs = model(user_ids, item_ids)
        # print("outputs shape:", outputs.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        userIndex +=1
        print(f"Batch loss: {loss.item()} for user {userIndex}")
        total_loss += loss.item()
    return total_loss / len(dataloader)

# 评估函数
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    index = 0
    with torch.no_grad():
        print("Starting evaluation...")
        for user_ids, item_ids, labels, _ in dataloader:
            index += 1
            user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
            outputs = model(user_ids, item_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            print(f"evaluation Batch loss: {loss.item()} for user index  {index}")
    return total_loss / len(dataloader)

# 训练循环
N_EPOCHS = 10
for epoch in range(N_EPOCHS):
    print(f"Epoch {epoch + 1}/{N_EPOCHS}")
    train_loss = train(model, train_loader, optimizer, criterion)
    test_loss = evaluate(model, test_loader, criterion)
    print(f'Epoch {epoch + 1}/{N_EPOCHS}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

print("Training finished.")
