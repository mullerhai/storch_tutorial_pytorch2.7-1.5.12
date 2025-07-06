import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义 Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, src):
        src = self.embedding(src)
        output = self.transformer_encoder(src)
        output = self.fc(output)
        return output

# 生成一些示例数据
input_dim = 10
output_dim = 10
d_model = 128
nhead = 4
num_layers = 2
seq_len = 5
batch_size = 32
num_samples = 1000

# 生成随机输入和目标序列
input_seqs = torch.randint(0, input_dim, (num_samples, seq_len))
target_seqs = torch.randint(0, output_dim, (num_samples, seq_len))

# 创建数据集和数据加载器
dataset = data.TensorDataset(input_seqs, target_seqs)
dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型、损失函数和优化器
model = TransformerModel(input_dim, d_model, nhead, num_layers, output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.view(-1, output_dim)
        targets = targets.view(-1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

print("Training finished.")
