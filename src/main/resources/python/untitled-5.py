import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data, datasets
import math

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义字段
TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm', lower=True, batch_first=True)
LABEL = data.LabelField(dtype=torch.float)

# 加载 IMDB 数据集
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

# 划分验证集
train_data, valid_data = train_data.split()

# 构建词汇表
TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d")
LABEL.build_vocab(train_data)

# 创建迭代器
BATCH_SIZE = 64
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device
)

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

# 模型参数
VOCAB_SIZE = len(TEXT.vocab)
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
NUM_CLASSES = 1
NUM_EXPERTS = 4
DROPOUT = 0.5

# 初始化模型、损失函数和优化器
model = TransformerMoEClassifier(VOCAB_SIZE, D_MODEL, NHEAD, NUM_LAYERS, NUM_CLASSES, NUM_EXPERTS, DROPOUT).to(device)

# 加载预训练词向量
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text)
        loss = criterion(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# 评估模型
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text)
            loss = criterion(predictions, batch.label)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# 训练循环
N_EPOCHS = 5
for epoch in range(N_EPOCHS):
    train_loss = train(model, train_iterator, optimizer, criterion)
    valid_loss = evaluate(model, valid_iterator, criterion)
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Val. Loss: {valid_loss:.3f}')

# 测试模型
test_loss = evaluate(model, test_iterator, criterion)
print(f'Test Loss: {test_loss:.3f}')

