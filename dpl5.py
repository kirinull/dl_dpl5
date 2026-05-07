# -*- coding: utf-8 -*-
"""
实验五 模型训练与测试 - 猫狗分类
完全基于实验四 CNN 结构（conv->relu->pool->flatten->fc），优化深度和宽度，增加正则化
目标：验证集准确率 > 90%
修复 Windows 多进程错误：使用 if __name__ == "__main__" 保护主代码
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# -------------------- 设置随机种子 --------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -------------------- 优化的 CNN 模型（严格遵循 conv->relu->pool->flatten->fc）--------------------
class OptimizedCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(OptimizedCNN, self).__init__()
        # 第一个卷积块: 3 -> 32, 输入 128x128 -> 64x64
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)

        # 第二个卷积块: 32 -> 64, 64x64 -> 32x32+--
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)

        # 第三个卷积块: 64 -> 128, 32x32 -> 16x16
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, 2)

        # 第四个卷积块: 128 -> 256, 16x16 -> 8x8
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, 2)

        # 展平
        self.flatten = nn.Flatten()
        # 全连接部分
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


# -------------------- 主程序入口（解决 Windows 多进程错误）--------------------
if __name__ == "__main__":
    # 设置随机种子
    set_seed(42)

    # -------------------- 数据集路径 --------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(script_dir, 'cat-dog_data')
    train_path = os.path.join(data_root, 'train')
    val_path = os.path.join(data_root, 'validation')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"训练集路径不存在: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"验证集路径不存在: {val_path}")

    # -------------------- 数据预处理 --------------------
    input_size = 128
    train_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # -------------------- 加载数据集 --------------------
    train_dataset = datasets.ImageFolder(train_path, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_path, transform=val_transforms)

    classes = train_dataset.classes
    num_classes = len(classes)
    print(f"类别: {classes}，共 {num_classes} 类")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")

    sample_img, sample_label = train_dataset[0]
    print(f"单张图片形状: {sample_img.shape} (C, H, W)")
    print(f"示例标签: {sample_label} -> {classes[sample_label]}")

    # -------------------- DataLoader（Windows 下可安全使用 num_workers）--------------------
    batch_size = 64  # 可根据 GPU 显存调整
    num_workers = 4  # 多进程加载，已用 if __name__ 保护
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # -------------------- 设备 --------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # -------------------- 模型、损失函数、优化器、调度器 --------------------
    model = OptimizedCNN(num_classes=num_classes).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}, 可训练参数量: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

    # -------------------- 训练配置 --------------------
    num_epochs = 80
    train_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    patience_counter = 0
    early_stop_patience = 10
    best_model_path = 'best_catdog_cnn.pth'

    print("\n===== 开始训练（优化的 CNN，符合实验四结构）=====")
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # ---------- 训练 ----------
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (preds == labels).sum().item()

        epoch_loss = running_loss / len(train_dataset)
        train_acc = 100.0 * correct_train / total_train
        train_losses.append(epoch_loss)

        # ---------- 验证 ----------
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (preds == labels).sum().item()
        val_acc = 100.0 * correct_val / total_val
        val_accuracies.append(val_acc)

        # 学习率调整
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']

        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch:2d}/{num_epochs}] | Loss: {epoch_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.6f} | Time: {epoch_time:.2f}s")

        # 保存最佳模型及早停
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> ★ 保存最佳模型 (准确率: {best_val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\n早停触发！连续 {early_stop_patience} 个 epoch 验证准确率未提升。")
                print(f"最佳验证准确率: {best_val_acc:.2f}%")
                break

    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"\n训练完成！最佳验证准确率: {best_val_acc:.2f}%")
    if best_val_acc >= 90:
        print("🎉 满足实验要求：验证集准确率 > 90%")
    else:
        print("⚠️ 准确率未达90%，建议增加训练轮次或调整数据增强强度。")

    # -------------------- 最终测试（验证集）--------------------
    model.eval()
    correct_test = 0
    total_test = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    final_acc = 100.0 * correct_test / total_test
    print(f"\n最终验证集准确率: {final_acc:.2f}% ({correct_test}/{total_test})")

    # -------------------- 可视化训练曲线 --------------------
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, 'r-s', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy Curve')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_curves_cnn.png')
    plt.show()
    print("训练曲线已保存为 training_curves_cnn.png")

    # -------------------- 混淆矩阵 --------------------
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title('Confusion Matrix on Validation Set')
    plt.tight_layout()
    plt.savefig('confusion_matrix_cnn.png')
    plt.show()

    # -------------------- 模型保存与加载验证 --------------------
    loaded_model = OptimizedCNN(num_classes=num_classes).to(device)
    loaded_model.load_state_dict(torch.load(best_model_path, map_location=device))
    loaded_model.eval()

    test_img, test_label = val_dataset[0]
    test_tensor = test_img.unsqueeze(0).to(device)
    with torch.no_grad():
        output = loaded_model(test_tensor)
        _, pred_cls = torch.max(output, 1)
    print(f"\n模型加载验证：真实标签 = {classes[test_label]} (ID: {test_label}), "
          f"预测标签 = {classes[pred_cls.item()]} (ID: {pred_cls.item()})")
    if test_label == pred_cls.item():
        print("✅ 加载模型推理正确，模型保存与加载功能正常。")
    else:
        print("⚠️ 此单样本预测错误，但整体准确率可能仍然很高。")

    # 清理显存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print("GPU 显存已清理。")