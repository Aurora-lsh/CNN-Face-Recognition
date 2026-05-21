# 🌟 明星人脸识别系统 (Star Face Recognition)

基于 PyTorch 与 ResNet18 迁移学习构建的 32 分类明星人脸识别项目。包含模型训练、命令行推理、Gradio Web 交互界面、OpenCV 实时摄像头检测以及完整的数据评估脚本。

## 🎯 项目简介
本项目使用深度学习技术（CNN）实现了对 32 位当红明星的人脸识别。项目采用了预训练的 **ResNet18** 作为主干网络进行迁移学习，并结合 **MTCNN** 实现了自动化的人脸检测与裁剪，最终在测试集上达到了 **73.55%** 的准确率。

## ✨ 核心功能
- **🚀 高效训练**：基于 ResNet18 迁移学习，快速收敛，支持 GPU 加速。
- **📸 实时追踪**：使用 OpenCV + MTCNN 实现实时摄像头人脸捕获与置信度显示。
- **🌐 Web 交互**：基于 Gradio 搭建的现代化网页 UI，支持图片拖拽与在线拍照识别。
- **📊 科学评估**：自动计算 Accuracy、Precision、Recall 等指标，并生成热力图混淆矩阵。

## 🛠️ 环境依赖
建议使用 Python 3.8 及以上版本。请在终端运行以下命令安装依赖：

```bash
# 1. 安装 PyTorch (请根据你的 CUDA 版本调整，此处以 CUDA 12.1 为例)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# 2. 安装其他核心依赖包
pip install facenet-pytorch opencv-python gradio scikit-learn matplotlib seaborn pandas
```
## 📂 目录结构
```
CNN-Face-Project/
├── dataset_split/          # 划分好的数据集 (需自行准备)
│   ├── train/              # 训练集 (包含32个明星的子文件夹)
│   └── val/                # 验证/测试集
├── best_face_model.pth     # 训练好的模型权重 (需自行训练或下载)
├── requirements.txt        # 项目所需的库依赖清单
├── train.py                # 模型训练脚本
├── app.py                  # 命令行图片识别应用
├── gui_app.py              # Gradio Web 网页应用
├── realtime_app.py         # OpenCV 摄像头实时识别应用
└── evaluate.py             # 模型评估与混淆矩阵生成脚本
```
## 🚀 快速开始
1. 模型训练
   数据集来源于教师发的实验素材，可在飞桨或kaggle下载类似数据集。使用自己的数据集，请按下述目录结构放置。
dataset_split/          # 划分好的数据集 (需自行准备)
  ├── train/              # 训练集 (包含32个明星的子文件夹)
  │    ├── label01
  │    ├── label01
  │    └── ······
  └── val/                # 验证/测试集
       ├── label01
       ├── label01
       └── ······
运行：
```bash
python train.py
```
2. 启动 Web 网页端
带有漂亮 UI 进度条的交互界面，支持上传照片：
```bash
python gui_app.py
```
3. 启动实时摄像头检测
调用本地摄像头，实时框出人脸并显示姓名与确信度（按 q 键退出）：
```bash
python realtime_app.py
```

4. 模型性能评估
对验证集进行全面测试，并生成 confusion_matrix.png：
```bash
python evaluate.py
```

## 实例结果
<img width="960" height="719" alt="结果" src="https://github.com/user-attachments/assets/6ac8cfe1-5ab6-4c8d-be30-bc1a86e44216" />


⚠️ 注意事项
由于 GitHub 的文件大小限制，本仓库未包含原始数据集和超过 100MB 的模型权重文件 (best_face_model.pth)。

如果需要直接运行推理脚本，请先执行 train.py 训练你自己的模型，或联系作者获取预训练权重。
