import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据加载与预处理
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    # 假设前 20 列是特征，第 21 列是技能 ID，第 22 列是是否正确，第 23 列是时间戳，第 24 列是耗时
    feature_columns = data.columns[:20]
    skill_column = data.columns[20]
    correct_column = data.columns[21]
    timestamp_column = data.columns[22]
    duration_column = data.columns[23]

    user_ids = data['user_id'].unique()
    sequences = []
    for user_id in user_ids:
        user_data = data[data['user_id'] == user_id]
        user_data = user_data.sort_values(by=timestamp_column)

        features = user_data[feature_columns].values
        skills = user_data[skill_column].values
        corrects = user_data[correct_column].values
        timestamps = user_data[timestamp_column].values
        durations = user_data[duration_column].values

        sequence = np.hstack([features, skills.reshape(-1, 1), corrects.reshape(-1, 1),
                              timestamps.reshape(-1, 1), durations.reshape(-1, 1)])
        sequences.append(sequence)

    # 填充序列
    max_length = max([len(seq) for seq in sequences])
    padded_sequences = []
    for seq in sequences:
        padded_seq = np.pad(seq, ((0, max_length - len(seq)), (0, 0)), 'constant')
        padded_sequences.append(padded_seq)

    return np.array(padded_sequences)

# 自定义数据集类
class KnowledgeTracingDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        features = seq[:, :20]
        skills = seq[:, 20].long()
        corrects = seq[:, 21].long()
        timestamps = seq[:, 22]
        durations = seq[:, 23]
        return features, skills, corrects, timestamps, durations

# 定义 IRT 模块
class IRTModule(nn.Module):
    def __init__(self, num_skills):
        super(IRTModule, self).__init__()
        self.ability = nn.Embedding(1, 1)  # 全局能力参数
        self.difficulty = nn.Embedding(num_skills, 1)
        self.discrimination = nn.Embedding(num_skills, 1)

    def forward(self, skills):
        ability = self.ability(torch.zeros(skills.size(0), dtype=torch.long, device=device))
        difficulty = self.difficulty(skills)
        discrimination = self.discrimination(skills)
        logits = discrimination * (ability - difficulty)
        prob = torch.sigmoid(logits)
        return prob

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

# 定义知识追踪模型
class KnowledgeTracingModel(nn.Module):
    def __init__(self, num_skills, input_dim, d_model, num_experts, d_ff, nhead, num_layers, dropout):
        super(KnowledgeTracingModel, self).__init__()
        self.skill_embedding = nn.Embedding(num_skills, input_dim // 4)
        self.feature_proj = nn.Linear(20, input_dim // 4)
        self.time_embedding = nn.Linear(1, input_dim // 4)
        self.duration_embedding = nn.Linear(1, input_dim // 4)
        self.irt_module = IRTModule(num_skills)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(input_dim, d_model, num_experts, d_ff, nhead, dropout) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features, skills, timestamps, durations):
        feature_embed = self.feature_proj(features)
        skill_embed = self.skill_embedding(skills)
        time_embed = self.time_embedding(timestamps.unsqueeze(-1))
        duration_embed = self.duration_embedding(durations.unsqueeze(-1))

        x = torch.cat([feature_embed, skill_embed, time_embed, duration_embed], dim=-1)

        irt_prob = self.irt_module(skills)

        for block in self.transformer_blocks:
            x = block(x)

        output = self.fc(x)
        output = self.sigmoid(output).squeeze(-1)

        # 结合 IRT 概率
        combined_output = 0.5 * output + 0.5 * irt_prob.squeeze(-1)
        return combined_output

# 训练函数
def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    for features, skills, corrects, timestamps, durations in train_loader:
        features, skills, corrects, timestamps, durations = features.to(device), skills.to(device), corrects.to(device), timestamps.to(device), durations.to(device)
        inputs_features = features[:, :-1]
        inputs_skills = skills[:, :-1]
        inputs_timestamps = timestamps[:, :-1]
        inputs_durations = durations[:, :-1]
        targets = corrects[:, 1:].float()

        optimizer.zero_grad()
        outputs = model(inputs_features, inputs_skills, inputs_timestamps, inputs_durations)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')

# 主函数
def main():
    file_path = 'your_dataset.csv'  # 替换为实际的数据文件路径
    sequences = load_and_preprocess_data(file_path)
    num_skills = int(sequences[:, :, 20].max()) + 1

    train_sequences, test_sequences = train_test_split(sequences, test_size=0.2, random_state=42)

    train_dataset = KnowledgeTracingDataset(train_sequences)
    test_dataset = KnowledgeTracingDataset(test_sequences)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 模型参数
    input_dim = 128
    d_model = 256
    num_experts = 4
    d_ff = 512
    nhead = 4
    num_layers = 2
    dropout = 0.1

    model = KnowledgeTracingModel(num_skills, input_dim, d_model, num_experts, d_ff, nhead, num_layers, dropout).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        train(model, device, train_loader, criterion, optimizer, epoch)

if __name__ == "__main__":
    main()
