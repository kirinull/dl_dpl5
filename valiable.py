# -*- coding: utf-8 -*-
"""
实验五（七）模型保存与加载验证
- 保存训练好的模型参数（.pth）
- 重新加载模型
- 验证可正常推理（对比加载前后输出一致性）
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


# ---------- 1. 定义模型结构（必须与训练时完全一致）----------
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


# ---------- 2. 保存模型参数 ----------
def save_model(model, filepath='best_catdog_cnn.pth'):
    torch.save(model.state_dict(), filepath)
    print(f"模型参数已保存至 {filepath}")


# ---------- 3. 加载模型参数 ----------
def load_model(model, filepath='best_catdog_cnn.pth', device='cpu'):
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    print(f"模型参数已从 {filepath} 加载")
    return model


# ---------- 4. 验证推理一致性 ----------
def verify_inference(original_model, loaded_model, test_loader, device):
    """
    验证原始模型与加载后的模型在相同输入上的输出是否一致
    """
    original_model.eval()
    loaded_model.eval()

    # 取一个batch的数据
    images, labels = next(iter(test_loader))
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        orig_outputs = original_model(images)
        load_outputs = loaded_model(images)

    # 比较输出是否完全一致
    is_equal = torch.allclose(orig_outputs, load_outputs, atol=1e-6)
    max_diff = torch.max(torch.abs(orig_outputs - load_outputs)).item()

    print(f"\n===== 推理一致性验证 =====")
    print(f"输出张量是否完全一致: {is_equal}")
    print(f"最大绝对差异: {max_diff:.8f}")

    # 进一步验证预测类别是否一致
    _, orig_pred = torch.max(orig_outputs, 1)
    _, load_pred = torch.max(load_outputs, 1)
    pred_match = (orig_pred == load_pred).all().item()
    print(f"预测类别是否完全一致: {pred_match}")

    # 展示前5个样本的预测结果
    print("\n前5个样本的预测对比：")
    for i in range(min(5, len(orig_pred))):
        print(f"  样本{i + 1}: 原始模型预测={orig_pred[i].item()}, 加载模型预测={load_pred[i].item()}, "
              f"真实标签={labels[i].item()}")


# ---------- 主程序 ----------
if __name__ == "__main__":
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 准备测试数据（只需少量样本验证推理）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(script_dir, 'cat-dog_data')
    val_path = os.path.join(data_root, 'validation')

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = datasets.ImageFolder(val_path, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    # 创建原始模型（训练好的）
    original_model = OptimizedCNN(num_classes=2).to(device)
    # 假设已经训练并保存了模型，这里直接加载（实际训练时已保存）
    model_path = 'best_catdog_cnn.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在，请先训练模型并保存。")
    original_model.load_state_dict(torch.load(model_path, map_location=device))
    print("原始模型加载完成。")

    # 保存模型参数（示范保存操作）
    save_model(original_model, model_path)

    # 重新创建新模型实例并加载参数
    loaded_model = OptimizedCNN(num_classes=2)
    loaded_model = load_model(loaded_model, model_path, device)

    # 验证推理一致性
    verify_inference(original_model, loaded_model, val_loader, device)