#该代码是一个完整的实时人脸识别系统，使用了预训练的 ResNet18 模型进行人脸识别，并结合 MTCNN 进行人脸检测。
# 它能够从摄像头实时捕捉视频流，检测画面中的人脸，并在每个检测到的人脸上画出框和显示识别结果（名字和置信度）。
import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from facenet_pytorch import MTCNN

# ================= 1. 初始化 AI 模型 =================
print("🚀 正在启动实时视觉引擎...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_weights_path = 'best_face_model.pth'
train_dir = './dataset_split/train'

# 获取类别名称
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"找不到训练集路径 {train_dir}")
class_names = sorted(os.listdir(train_dir))
num_classes = len(class_names)

# 加载 ResNet18 识别模型
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load(model_weights_path, map_location=device))
model = model.to(device)
model.eval()

# 图像预处理 (必须和训练时一致)
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载 MTCNN 人脸检测器 (keep_all=True 允许同时检测画面里的多个人)
mtcnn = MTCNN(keep_all=True, device=device)

# ================= 2. 解决 OpenCV 中文乱码的工具函数 =================
def put_chinese_text(img_cv2, text, position, text_color=(0, 255, 0)):
    """将 OpenCV 图片转为 PIL 格式，画上中文后再转回来"""
    img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 尝试加载 Windows 默认的微软雅黑字体
    try:
        font = ImageFont.truetype("msyh.ttc", 20)
    except IOError:
        # 如果找不到微软雅黑，回退到默认字体(可能无法显示中文)
        font = ImageFont.load_default()
        
    # 在图片上绘制文字
    draw.text(position, text, font=font, fill=text_color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ================= 3. 开启摄像头实时检测 =================
cap = cv2.VideoCapture(0) # 0 表示默认摄像头

if not cap.isOpened():
    print("❌ 无法打开摄像头，请检查设备是否连接或被其他程序占用。")
    exit()

print("✅ 摄像头已开启！按键盘上的 'q' 键退出程序。")

while True:
    # 1. 读取摄像头的一帧画面
    ret, frame = cap.read()
    if not ret:
        break
        
    # 画面水平翻转，像照镜子一样，体验更好
    frame = cv2.flip(frame, 1)

    # OpenCV 默认是 BGR 格式，转换为 PIL 需要的 RGB 格式
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    # 2. 检测人脸
    with torch.no_grad():
        boxes, _ = mtcnn.detect(pil_img)

    # 3. 如果画面里有脸，开始识别并画框
    if boxes is not None:
        for box in boxes:
            # 获取人脸坐标并增加边缘 (Margin)
            margin = 15
            x1, y1, x2, y2 = [int(b) for b in box]
            x1_crop = max(0, x1 - margin)
            y1_crop = max(0, y1 - margin)
            x2_crop = min(pil_img.width, x2 + margin)
            y2_crop = min(pil_img.height, y2 + margin)

            # 抠出人脸并预处理
            face_img = pil_img.crop((x1_crop, y1_crop, x2_crop, y2_crop))
            input_tensor = data_transform(face_img).unsqueeze(0).to(device)

            # 模型预测是谁
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, 0)
                
            predicted_name = class_names[predicted_idx.item()]
            conf_percent = confidence.item() * 100
            
            # 准备要显示的文字
            display_text = f"{predicted_name} {conf_percent:.1f}%"
            
            # 根据置信度决定框和文字的颜色 (确信度>70%为绿色，否则为黄色预警)
            if conf_percent > 70.0:
                color = (0, 255, 0) # 绿色 BGR
                text_color = (0, 255, 0) # 绿色 RGB for PIL
            else:
                color = (0, 255, 255) # 黄色 BGR
                text_color = (255, 255, 0) # 黄色 RGB for PIL

            # 在 OpenCV 画面上画长方形框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 在框的左上方写上名字和置信度
            text_position = (x1, max(0, y1 - 30))
            frame = put_chinese_text(frame, display_text, text_position, text_color)

    # 4. 显示最终画面
    cv2.imshow('Live Face Recognition', frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源并关闭窗口
cap.release()
cv2.destroyAllWindows()