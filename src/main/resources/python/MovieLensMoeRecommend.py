import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import urllib.request
import zipfile

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 下载并解压 MovieLens 1M 数据集
DATA_URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = "ml-1m"
DATA_PATH = os.path.join(DATA_DIR, "ratings.dat")

if not os.path.exists(DATA_DIR):
    print("Downloading MovieLens 1M dataset...")
    urllib.request.urlretrieve(DATA_URL, "ml-1m.zip")
    print("Extracting dataset...")
    with zipfile.ZipFile("ml-1m.zip", 'r') as zip_ref:
        zip_ref.extractall(".")
    os.remove("ml-1m.zip")

# 读取数据
print("pandas Reading data...")
df = pd.read_csv(DATA_PATH, sep="::", engine='python', header=None,
                 names=['userId', 'movieId', 'rating', 'timestamp'])

# 对用户 ID 和电影 ID 进行编码
user_ids = df['userId'].unique()
movie_ids = df['movieId'].unique()

user_id_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
movie_id_map = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

df['userId'] = df['userId'].map(user_id_map)
df['movieId'] = df['movieId'].map(movie_id_map)

print("Data loaded train_test_split .")
# 划分训练集和测试集
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)


# 定义数据集类
class MovieLensDataset(Dataset):
    def __init__(self, data):
        self.user_ids = torch.tensor(data['userId'].values, dtype=torch.long)
        self.movie_ids = torch.tensor(data['movieId'].values, dtype=torch.long)
        self.ratings = torch.tensor(data['rating'].values, dtype=torch.float)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.movie_ids[idx], self.ratings[idx]


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
    def __init__(self, num_users, num_movies, d_model, nhead, num_layers, num_experts, dropout):
        super(TransformerMoERecommender, self).__init__()
        self.user_embedding = nn.Embedding(num_users, d_model)
        self.movie_embedding = nn.Embedding(num_movies, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout),
            num_layers=num_layers
        )
        self.moe = MoE(num_experts, d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, user_ids, movie_ids):
        user_embed = self.user_embedding(user_ids)
        movie_embed = self.movie_embedding(movie_ids)
        combined_embed = user_embed + movie_embed
        combined_embed = combined_embed.unsqueeze(0)  # 添加序列维度
        output = self.transformer_encoder(combined_embed)
        output = output.squeeze(0)
        output = self.moe(output)
        return output.squeeze()


# 初始化数据集和数据加载器
print("Initializing dataset and dataloader...")
train_dataset = MovieLensDataset(train_data)
test_dataset = MovieLensDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print("Dataset and dataloader initialized successfully.")
# 模型参数
NUM_USERS = len(user_ids)
NUM_MOVIES = len(movie_ids)
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
NUM_EXPERTS = 4
DROPOUT = 0.5

# 初始化模型、损失函数和优化器
model = TransformerMoERecommender(NUM_USERS, NUM_MOVIES, D_MODEL, NHEAD, NUM_LAYERS, NUM_EXPERTS, DROPOUT).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练函数
def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    print("Training model...")
    for user_ids, movie_ids, ratings in dataloader:
        user_ids, movie_ids, ratings = user_ids.to(device), movie_ids.to(device), ratings.to(device)
        optimizer.zero_grad()
        outputs = model(user_ids, movie_ids)
        loss = criterion(outputs, ratings)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


# 评估函数
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        print("Evaluating model...")
        for user_ids, movie_ids, ratings in dataloader:
            user_ids, movie_ids, ratings = user_ids.to(device), movie_ids.to(device), ratings.to(device)
            outputs = model(user_ids, movie_ids)
            loss = criterion(outputs, ratings)
            total_loss += loss.item()
    return total_loss / len(dataloader)


# 训练循环
N_EPOCHS = 10
for epoch in range(N_EPOCHS):
    print(f"Epoch {epoch + 1}/{N_EPOCHS}")
    train_loss = train(model, train_loader, optimizer, criterion)
    test_loss = evaluate(model, test_loader, criterion)
    print(f'Epoch {epoch + 1}/{N_EPOCHS}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

print("Training finished.")
