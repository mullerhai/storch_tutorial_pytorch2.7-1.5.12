import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载 MNIST 数据集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                           download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 定义 Transformer 模型
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, num_classes):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers
        )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, src):
        src = src.view(src.size(0), -1, 1).squeeze(-1)  # 调整输入形状为 [batch_size, seq_len]
        src = self.embedding(src)
        src = src.permute(1, 0, 2)  # 调整为 [seq_len, batch_size, d_model]
        memory = self.transformer_encoder(src)
        tgt = torch.zeros_like(memory)  # 简单的目标输入
        output = self.transformer_decoder(tgt, memory)
        output = output.mean(dim=0)  # 对序列维度求平均
        output = self.fc(output)
        return output

# 模型参数
input_dim = 28 * 28  # MNIST 图像大小
d_model = 128
nhead = 4
num_layers = 2
num_classes = 10

# 初始化模型、损失函数和优化器
model = TransformerClassifier(input_dim, d_model, nhead, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        images = images.view(images.size(0), -1)  # 展平图像

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

print("Training finished.")
