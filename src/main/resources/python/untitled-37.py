import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 下载并加载 LIBRISPEECH 数据集
train_dataset = torchaudio.datasets.LIBRISPEECH(
    root=".\\data",
    url="train-clean-100",
    download=True
)
test_dataset = torchaudio.datasets.LIBRISPEECH(
    root=".\\data",
    url="test-clean",
    download=True
)

# 定义字符映射
char_map_str = """
' 0
<SPACE> 1
a 2
b 3
c 4
d 5
e 6
f 7
g 8
h 9
i 10
j 11
k 12
l 13
m 14
n 15
o 16
p 17
q 18
r 19
s 20
t 21
u 22
v 23
w 24
x 25
y 26
z 27
"""
char_map = {}
index_map = {}
for line in char_map_str.strip().split('\n'):
    ch, index = line.split()
    char_map[ch] = int(index)
    index_map[int(index)] = ch
index_map[1] = ' '

# 数据预处理函数
def data_processing(data, char_map):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (waveform, _, utterance, _, _, _) in data:
        # 减小 n_mels 的值
        spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=64
        )(waveform).squeeze(0).transpose(0, 1)
        spectrograms.append(spectrogram)
        label = torch.tensor([char_map.get(c, char_map['<SPACE>']) for c in utterance.lower().replace(' ', '<SPACE>')])
        labels.append(label)
        input_lengths.append(spectrogram.shape[0] // 2)
        label_lengths.append(len(label))

    spectrograms = pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = pad_sequence(labels, batch_first=True)
    return spectrograms, labels, input_lengths, label_lengths

# 定义专家网络
class Expert(nn.Module):
    def __init__(self, input_dim, d_model):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, d_model)
        self.fc2 = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义门控网络
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return nn.functional.softmax(self.fc(x), dim=1)

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
        # 调整维度扩展逻辑
        gates = gates.unsqueeze(1).unsqueeze(-1)
        # 确保扩展维度与 expert_outputs 匹配
        print(f"gates shape: {gates.shape}, expert_outputs shape: {expert_outputs.shape}")
        gates = gates.expand(-1, expert_outputs.size(1), expert_outputs.size(2), expert_outputs.size(3))
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

# 定义语音识别模型
class SpeechRecognitionModel(nn.Module):
    def __init__(self, input_dim, d_model, num_experts, d_ff, nhead, num_layers, num_classes, dropout):
        super(SpeechRecognitionModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, d_model, num_experts, d_ff, nhead, dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.squeeze(1).transpose(1, 2)
        x = self.embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.fc(x)
        x = x.transpose(0, 1)
        return x

# 模型参数
# 根据修改后的 n_mels 调整 input_dim
input_dim = 64
d_model = 256
num_experts = 4
d_ff = 1024
nhead = 4
num_layers = 2
num_classes = len(char_map)
dropout = 0.1

# 初始化模型、损失函数和优化器
model = SpeechRecognitionModel(
    input_dim, d_model, num_experts, d_ff, nhead, num_layers, num_classes, dropout
).to(device)
criterion = nn.CTCLoss(blank=0)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练函数
def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    data_len = len(train_loader.dataset)
    for batch_idx, _data in enumerate(train_loader):
        spectrograms, labels, input_lengths, label_lengths = _data
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model(spectrograms)
        output = nn.functional.log_softmax(output, dim=2)
        loss = criterion(output, labels, input_lengths, label_lengths)
        loss.backward()

        optimizer.step()
        if batch_idx % 100 == 0 or batch_idx == data_len:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(spectrograms), data_len,
                100. * batch_idx / len(train_loader), loss.item()))

# 测试函数
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, _data in enumerate(test_loader):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            output = model(spectrograms)
            output = nn.functional.log_softmax(output, dim=2)
            loss = criterion(output, labels, input_lengths, label_lengths)
            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}\n'.format(test_loss))

# 数据加载器
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=20,
    shuffle=True,
    collate_fn=lambda x: data_processing(x, char_map)
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=20,
    shuffle=False,
    collate_fn=lambda x: data_processing(x, char_map)
)

print("train_loader len : ",len(train_loader))
print("test_loader len : ",len(test_loader))
# 训练和测试
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, criterion, optimizer, epoch)
    test(model, device, test_loader, criterion)
