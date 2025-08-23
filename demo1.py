# 加载预训练模型
# 安装三方库：ultralytics

# 导包：
from ultralytics import YOLO

# 1、加载模型
model = YOLO("yolov8n.pt")

# 2、目标检测
results = model("test.png", show=True, save=True, project="runs/detect")