import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import requests
import zipfile
import numpy as np
import re

# 下载并解压数据集
def download_and_extract(url, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    zip_path = os.path.join(save_path, 'license_plate_dataset.zip')
    if not os.path.exists(zip_path):
        print("Downloading dataset...")
        response = requests.get(url)
        with open(zip_path, 'wb') as f:
            f.write(response.content)
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(save_path)


# 自定义数据集类
class LicensePlateDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.labels = []

        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(os.path.join(root, file))
                    label = re.search(r'([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼]{1}[A-HJ-NP-Z]{1}[A-HJ-NP-Z0-9]{5})', file)
                    if label:
                        self.labels.append(label.group(1))
                    else:
                        self.labels.append('')

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        # 简单的标签编码
        char_to_idx = {char: i for i, char in enumerate('京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼ABCDEFGHJKLMNPQRSTUVWXYZ0123456789')}
        label_encoded = [char_to_idx[char] for char in label]
        label_encoded = torch.tensor(label_encoded, dtype=torch.long)

        return image, label_encoded


# 残差卷积块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


# 专家网络
class Expert(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Expert, self).__init__()
        self.residual_blocks = nn.Sequential(
            ResidualBlock(in_channels, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2)
        )
        self.fc = nn.Linear(256 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.residual_blocks(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 门控网络
class GatingNetwork(nn.Module):
    def __init__(self, in_channels, num_experts):
        super(GatingNetwork, self).__init__()
        self.conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 16 * 16, num_experts)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x


# 混合专家模型
class MoE(nn.Module):
    def __init__(self, in_channels, num_classes, num_experts):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList([Expert(in_channels, num_classes) for _ in range(num_experts)])
        self.gating_network = GatingNetwork(in_channels, num_experts)

    def forward(self, x):
        gates = self.gating_network(x)
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        output = torch.sum(gates.unsqueeze(-1) * expert_outputs, dim=1)
        return output


# Transformer 编码器层
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


# 车牌识别模型
class LicensePlateRecognizer(nn.Module):
    def __init__(self, in_channels, num_classes, num_experts, d_model=256, nhead=4, num_layers=2):
        super(LicensePlateRecognizer, self).__init__()
        self.moe = MoE(in_channels, num_classes, num_experts)
        self.transformer_encoder = nn.Sequential(
            *[TransformerEncoderLayer(d_model, nhead) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        moe_output = self.moe(x)
        moe_output = moe_output.unsqueeze(0)  # 添加序列维度
        transformer_output = self.transformer_encoder(moe_output)
        transformer_output = transformer_output.squeeze(0)
        output = self.fc(transformer_output)
        return output


# 训练函数
def train(model, train_loader, criterion, optimizer, device):
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
    return total_loss / len(train_loader)


# 评估函数
def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(test_loader)


# 主函数
def main():
    # 下载并解压数据集（需要替换为实际数据集链接）
    url = 'https://example.com/license_plate_dataset.zip'
    save_path = 'data'
    download_and_extract(url, save_path)

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((64, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = LicensePlateDataset(root_dir=save_path, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    in_channels = 3
    num_classes = len('京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼ABCDEFGHJKLMNPQRSTUVWXYZ0123456789')
    num_experts = 4
    model = LicensePlateRecognizer(in_channels, num_classes, num_experts).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')


if __name__ == "__main__":
    main()


