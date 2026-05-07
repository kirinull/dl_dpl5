# -*- coding: utf-8 -*-
"""
模型测试与准确率计算（评估模式）
- 加载已训练模型
- 切换为 eval() 模式
- 关闭梯度计算 (torch.no_grad)
- 统计测试集上的正确预测数并计算准确率
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# -------------------- 定义模型结构（与训练时完全相同）--------------------
class OptimizedCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(OptimizedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# -------------------- 测试主程序 --------------------
if __name__ == "__main__":
    # 1. 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 2. 数据集路径（与训练时一致）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(script_dir, 'cat-dog_data')
    val_path = os.path.join(data_root, 'validation')   # 验证集/测试集

    # 3. 数据预处理（与验证时一致）
    input_size = 128
    val_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. 加载测试数据集
    test_dataset = datasets.ImageFolder(val_path, transform=val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    print(f"测试集样本数: {len(test_dataset)}")
    print(f"类别: {test_dataset.classes}")

    # 5. 加载训练好的模型
    model = OptimizedCNN(num_classes=2).to(device)
    model_path = "best_catdog_cnn.pth"   # 请确保该文件存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在，请先训练并保存模型。")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"成功加载模型: {model_path}")

    # -------------------- 核心：模型测试与准确率计算 --------------------
    # 5.1 切换为评估模式
    model.eval()

    # 5.2 关闭梯度计算
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 前向推理
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # 统计正确数
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 5.3 计算并输出准确率
    accuracy = 100.0 * correct / total
    print(f"\n===== 测试结果 =====")
    print(f"正确预测数: {correct} / {total}")
    print(f"测试准确率: {accuracy:.2f}%")

    if accuracy >= 90.0:
        print("🎉 满足实验要求：验证准确率 > 90%")
    else:
        print("⚠️ 当前准确率未达到90%，请尝试优化模型或调整超参数。")