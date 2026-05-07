# 该代码用于从原始数据集中裁剪人脸，并保存到新的目录中。它使用 MTCNN 模型进行人脸检测
# 并包含一些防御机制来处理可能出现的异常情况，如图片过大、过小或损坏等。
import os
import torch
from PIL import Image, ImageFile
from facenet_pytorch import MTCNN
from tqdm import tqdm

# 允许加载截断的损坏图片而不报错
ImageFile.LOAD_TRUNCATED_IMAGES = True

source_dir = r'./dataset'
target_dir = r'./dataset_cropped'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, device=device)

def crop_faces():
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for person_name in os.listdir(source_dir):
        person_path = os.path.join(source_dir, person_name)
        save_path = os.path.join(target_dir, person_name)
        
        if not os.path.isdir(person_path):
            continue
            
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 找出还没处理过的图片（断点续传逻辑）
        all_imgs = os.listdir(person_path)
        existing_imgs = set(os.listdir(save_path)) if os.path.exists(save_path) else set()
        pending_imgs = [img for img in all_imgs if img not in existing_imgs]
        
        if not pending_imgs:
            continue
            
        print(f"\n正在处理: {person_name} ({len(pending_imgs)}/{len(all_imgs)})")
        
        # 使用 tqdm 进度条
        pbar = tqdm(pending_imgs)
        for img_name in pbar:
            img_path = os.path.join(person_path, img_name)
            img_save_path = os.path.join(save_path, img_name)
            
            # 在进度条上显示当前正在处理的图片名，方便死机时定位
            pbar.set_description(f"Processing {img_name}")
            
            try:
                img = Image.open(img_path).convert('RGB')
                
                # 【防御机制 1】如果图片超大，等比例缩小它，防止显存(VRAM)溢出
                if img.width > 1920 or img.height > 1920:
                    img.thumbnail((1920, 1920))
                
                # 【防御机制 2】如果图片太小，直接跳过，MTCNN 提不出特征
                if img.width < 40 or img.height < 40:
                    tqdm.write(f"⚠️ 跳过 {img_name}: 图片尺寸过小 ({img.width}x{img.height})")
                    continue

                mtcnn(img, save_path=img_save_path)
                
            except Exception as e:
                # 【防御机制 3】捕获所有异常，打印错误但不退出
                tqdm.write(f"❌ 错误: 无法处理 {person_name}/{img_name} | 原因: {str(e)}")
                # 释放可能被卡住的显存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

if __name__ == '__main__':
    crop_faces()
    print(f"\n✅ 人脸裁剪全部完成！")