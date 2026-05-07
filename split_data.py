# 该代码用于将裁剪完成的干净人脸数据集划分为训练集和验证集。
# 它会遍历每个明星类别，随机打乱图片顺序，并按照设定的比例（70% 训练, 30% 验证）将图片复制到新的目录结构中。
import os
import random
import shutil
from tqdm import tqdm

# 1. 配置路径
source_dir = r'./dataset_cropped'      # 刚才裁剪完成的干净人脸数据集
output_dir = r'./dataset_split'        # 划分后保存的新主目录

train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')

# 2. 设置比例 (70% 训练, 30% 验证)
split_ratio = 0.7

def split_dataset():
    # 检查源文件夹是否存在
    if not os.path.exists(source_dir):
        print(f"找不到源文件夹: {source_dir}，请确认路径。")
        return

    # 遍历源目录下的每个明星类别
    classes = os.listdir(source_dir)
    
    for class_name in tqdm(classes, desc="正在按类别划分数据"):
        class_path = os.path.join(source_dir, class_name)
        
        if not os.path.isdir(class_path):
            continue
            
        # 为每个明星在 train 和 val 下分别创建同名文件夹
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        
        # 获取该明星的所有图片并打乱顺序
        images = os.listdir(class_path)
        # 过滤掉非图片文件
        images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)
        
        # 计算划分的索引点
        split_point = int(len(images) * split_ratio)
        
        train_images = images[:split_point]
        val_images = images[split_point:]
        
        # 复制文件到 train 文件夹
        for img in train_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(train_dir, class_name, img)
            shutil.copy(src, dst)
            
        # 复制文件到 val 文件夹
        for img in val_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(val_dir, class_name, img)
            shutil.copy(src, dst)

if __name__ == '__main__':
    print("开始打乱并划分数据集...")
    split_dataset()
    print(f"✅ 划分完成！数据集已保存至: {output_dir}")