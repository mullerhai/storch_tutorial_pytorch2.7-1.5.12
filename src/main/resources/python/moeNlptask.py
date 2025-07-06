import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.datasets.utils import download_and_extract_archive
import re
import spacy
from collections import Counter
import math

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 下载 IMDB 数据集
DATA_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATA_ROOT = "./data"
# download_and_extract_archive(DATA_URL, DATA_ROOT)
print("IMDB dataset downloaded and extracted.")
# 加载分词器
nlp = spacy.load("en_core_web_sm")
print("Spacy tokenizer loaded.")
# 定义数据集类
class IMDBDataset(Dataset):
    def __init__(self, root_dir, split='train', max_vocab_size=25000):
        self.root_dir = root_dir
        self.split = split
        self.data = []
        self.vocab = {}
        self.label_map = {'pos': 1, 'neg': 0}
        print("IMDBDataset dataset....")
        # 构建词汇表
        counter = Counter()
        data_dir = os.path.join(root_dir, 'aclImdb', split)
        for label in ['pos', 'neg']:
            print("construct dataset.... for label")
            index = 0
            label_dir = os.path.join(data_dir, label)
            for filename in os.listdir(label_dir):
                with open(os.path.join(label_dir, filename), 'r', encoding='utf-8') as f:
                    text = f.read()
                    index += 1
                    print(f"construct {label} dataset.... for text {index}  {text}")
                    tokens = [token.text.lower() for token in nlp(re.sub(r'<[^>]+>', '', text))]
                    counter.update(tokens)
                    if index >= 640:
                        break

        # 取出现频率最高的词构建词汇表
        vocab_list = [word for word, freq in counter.most_common(max_vocab_size - 2)]
        self.vocab = {'<pad>': 0, '<unk>': 1}
        self.vocab.update({word: idx + 2 for idx, word in enumerate(vocab_list)})
        print("construct dataset. complete ...")

        # 加载数据
        for label in ['pos', 'neg']:
            label_dir = os.path.join(data_dir, label)
            cntIndex = 0
            for filename in os.listdir(label_dir):
                with open(os.path.join(label_dir, filename), 'r', encoding='utf-8') as f:
                    text = f.read()
                    cntIndex += 1
                    print(f"load 22 {label} dataset.... for text cnt {cntIndex}  {text}")
                    tokens = [token.text.lower() for token in nlp(re.sub(r'<[^>]+>', '', text))]
                    token_ids = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
                    self.data.append((token_ids, self.label_map[label]))
                    if cntIndex >= 640:
                        break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 定义填充函数
def collate_fn(batch):
    texts, labels = zip(*batch)
    max_length = max(len(text) for text in texts)
    padded_texts = []
    index = 0
    for text in texts:
        index += 1
        print(f"collate_fn ... for text {index}  {text} len {len(text)} padded_texts {len(padded_texts)}")
        padded_text = text + [0] * (max_length - len(text))
        padded_texts.append(padded_text)
    return torch.tensor(padded_texts, dtype=torch.long).to(device), torch.tensor(labels, dtype=torch.float).to(device)

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

# 定义位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

# 定义 Transformer MoE 模型
class TransformerMoEClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, num_classes, num_experts, dropout):
        super(TransformerMoEClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout),
            num_layers=num_layers
        )
        self.moe = MoE(num_experts, d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src = self.embedding(src)
        src = self.dropout(src)
        src = self.positional_encoding(src)
        memory = self.transformer_encoder(src)
        memory = torch.mean(memory, dim=1)  # 对序列维度求平均
        output = self.moe(memory)
        return output.squeeze()

print("model init try to load dataset")
# 初始化数据集和数据加载器
train_dataset = IMDBDataset(DATA_ROOT, split='train')
test_dataset = IMDBDataset(DATA_ROOT, split='test', max_vocab_size=len(train_dataset.vocab))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

print("train_loader and test_loader initialized.")
# 模型参数
VOCAB_SIZE = len(train_dataset.vocab)
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
NUM_CLASSES = 1
NUM_EXPERTS = 4
DROPOUT = 0.5

# 初始化模型、损失函数和优化器
model = TransformerMoEClassifier(VOCAB_SIZE, D_MODEL, NHEAD, NUM_LAYERS, NUM_CLASSES, NUM_EXPERTS, DROPOUT).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    print("train function...")
    for texts, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# 评估函数
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    print("evaluate function...")
    with torch.no_grad():
        for texts, labels in dataloader:
            outputs = model(texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# 训练循环
N_EPOCHS = 50
for epoch in range(N_EPOCHS):
    print(f"Epoch {epoch + 1}/{N_EPOCHS}")
    train_loss = train(model, train_loader, optimizer, criterion)
    test_loss = evaluate(model, test_loader, criterion)
    print(f'Epoch: {epoch + 1:02}, Train Loss: {train_loss:.3f}, Test Loss: {test_loss:.3f}')
