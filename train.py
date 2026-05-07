#该代码用于训练一个基于 ResNet18 的人脸识别模型，使用之前准备好的裁剪好的人脸数据集。
# 它包含了数据增强、模型定义、训练循环以及自动保存最佳模型权重的功能。
# 训练完成后，最佳模型权重将被保存到本地文件中，以供后续评估和部署使用。
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import copy

# 1. 配置路径与参数
data_dir = './dataset_split'  # 刚才划分好的数据集路径
batch_size = 32
num_epochs = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🚀 正在使用计算设备: {device}")

# 2. 定义数据增强与预处理
# 针对已经裁剪好的纯净人脸，我们做一些轻量级的增强防止过拟合
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),    # 先放大一点点
        transforms.RandomCrop(224),       # 随机裁剪回224，增加位置多样性
        transforms.RandomHorizontalFlip(),# 随机水平翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # 随机调整亮度和对比度
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet 标准归一化
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),    # 验证集不加花哨的增强，直接缩放
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 3. 加载数据集
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ['train', 'val']
}

dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'), num_workers=4)
    for x in ['train', 'val']
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
num_classes = len(class_names)

print(f"📦 发现 {num_classes} 个明星类别。")
print(f"📊 训练集图片: {dataset_sizes['train']} 张 | 验证集图片: {dataset_sizes['val']} 张")

# 4. 初始化 ResNet18 迁移学习模型
# 使用预训练权重加速收敛
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 冻结前面的卷积层（可选，这里我们选择微调全量网络以获得更高精度）
# 修改最后一层全连接层，使其输出节点数为我们的明星数量 (32)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

# 定义损失函数 (交叉熵) 和优化器 (Adam)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# 5. 训练主循环
def train_model(model, criterion, optimizer, num_epochs=20):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 每个 epoch 都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置为训练模式
            else:
                model.eval()   # 设置为评估模式

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据批次
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad() # 清零梯度

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 只有在训练阶段才反向传播和更新参数
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计损失和准确率
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

            # 自动保存验证集准确率最高的模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"🌟 发现更好的模型！当前最高验证集准确率: {best_acc:.4f}")

    print(f'\n🎉 训练完成！最高验证集准确率: {best_acc:.4f}')
    
    # 恢复表现最好的参数
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    # 开始训练！
    best_model = train_model(model, criterion, optimizer, num_epochs=num_epochs)
    
    # 6. 保存最终的权重文件
    save_path = 'best_face_model.pth'
    torch.save(best_model.state_dict(), save_path)
    print(f"💾 最佳模型权重已保存至: {save_path}")