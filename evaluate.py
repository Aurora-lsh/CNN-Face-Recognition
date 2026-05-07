#该代码用于评估训练好的 ResNet18 模型在测试集上的性能，生成分类报告和混淆矩阵图。
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ================= 1. 解决 Matplotlib 中文显示问题 =================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei'] # 优先使用黑体或微软雅黑
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

def main():
    # ================= 2. 初始化配置 =================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_weights_path = 'best_face_model.pth'
    val_dir = './dataset_split/val'  # 使用30%的测试集进行评估

    # ================= 3. 数据加载与预处理 =================
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("📂 正在加载测试集数据...")
    val_dataset = datasets.ImageFolder(val_dir, data_transform)
    # 这里的 num_workers=4 在 Windows 下必须配合 if __name__ == '__main__' 使用
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    class_names = val_dataset.classes
    num_classes = len(class_names)

    # ================= 4. 加载训练好的模型 =================
    print("🚀 正在加载 ResNet18 模型...")
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model = model.to(device)
    model.eval()  # 设置为评估模式

    # ================= 5. 开始预测并收集结果 =================
    y_true = []
    y_pred = []

    print("⏳ 正在对测试集进行全面预测，请稍候...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # ================= 6. 计算指标并生成报告 =================
    print("\n" + "="*50)
    print("🎯 实验结果：分类性能报告 (Classification Report)")
    print("="*50)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    # ================= 7. 绘制并保存混淆矩阵图 =================
    print("📊 正在绘制混淆矩阵...")
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(16, 12)) 
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)

    plt.title('明星人脸识别 - 测试集混淆矩阵', fontsize=18)
    plt.xlabel('模型预测结果 (Predicted Label)', fontsize=14)
    plt.ylabel('真实明星身份 (True Label)', fontsize=14)
    plt.xticks(rotation=90) 
    plt.yticks(rotation=0)
    plt.tight_layout()

    # 保存图片到本地
    save_path = 'confusion_matrix.png'
    plt.savefig(save_path, dpi=300)
    print(f"✅ 混淆矩阵图已成功保存为: 【 {save_path} 】")
    plt.show() 

# 核心：Windows 下的多进程保护锁
if __name__ == '__main__':
    main()