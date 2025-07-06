import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import GloVe
import math

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义分词器
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

# 生成词汇表
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

train_iter = AG_NEWS(split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"], max_tokens=25000)
vocab.set_default_index(vocab["<unk>"])

# 加载预训练词向量
glove = GloVe(name='6B', dim=100)
pretrained_embeddings = glove.get_vecs_by_tokens(vocab.get_itos())

# 数据处理函数
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1  # AG News 标签从 1 开始，调整为从 0 开始

# 生成批次数据
def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.long).to(device)
    text_list = torch.cat(text_list).to(device)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0).to(device)
    return label_list, text_list, offsets

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
        self.embedding.weight.data.copy_(pretrained_embeddings)
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
        return output

# 模型参数
VOCAB_SIZE = len(vocab)
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
NUM_CLASSES = 4  # AG News 有 4 个类别
NUM_EXPERTS = 4
DROPOUT = 0.5

# 初始化模型、损失函数和优化器
model = TransformerMoEClassifier(VOCAB_SIZE, D_MODEL, NHEAD, NUM_LAYERS, NUM_CLASSES, NUM_EXPERTS, DROPOUT).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train(dataloader):
    model.train()
    total_loss = 0
    for labels, texts, offsets in dataloader:
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# 评估函数
def evaluate(dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for labels, texts, offsets in dataloader:
            outputs = model(texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# 加载数据集
train_iter = AG_NEWS(split='train')
test_iter = AG_NEWS(split='test')

# 创建数据加载器
BATCH_SIZE = 64
train_dataloader = DataLoader(list(train_iter), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(list(test_iter), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

# 训练循环
N_EPOCHS = 5
for epoch in range(N_EPOCHS):
    train_loss = train(train_dataloader)
    test_loss = evaluate(test_dataloader)
    print(f'Epoch: {epoch + 1:02}, Train Loss: {train_loss:.3f}, Test Loss: {test_loss:.3f}')
