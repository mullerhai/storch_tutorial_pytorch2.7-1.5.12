import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载 MNIST 数据集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                           download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                          download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义专家网络
class Expert(nn.Module):
    def __init__(self, input_dim, d_model):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, d_model)
        self.fc2 = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义门控网络
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=1)

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
    def __init__(self, input_dim, d_model, num_experts, d_ff, dropout):
        super(TransformerBlock, self).__init__()
        self.moe = MoE(input_dim, d_model, num_experts)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        moe_output = self.moe(x)
        x = self.norm1(x + self.dropout1(moe_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x

# 定义完整的 MoE Transformer 分类模型
class MoETransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model, num_experts, d_ff, num_layers, num_classes, dropout):
        super(MoETransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, d_model, num_experts, d_ff, dropout) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.embedding(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.fc(x)
        return x

# 模型参数
input_dim = 28 * 28
d_model = 128
num_experts = 4
d_ff = 512
num_layers = 2
num_classes = 10
dropout = 0.1

# 初始化模型、损失函数和优化器
model = MoETransformerClassifier(input_dim, d_model, num_experts, d_ff, num_layers, num_classes, dropout).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

print("Training finished.")

# 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')
