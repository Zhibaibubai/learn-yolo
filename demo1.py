# 加载预训练模型
# 安装三方库：ultralytics

# 导包：
from ultralytics import YOLO

# 1、加载模型
model = YOLO("yolov8n.pt")

# 2、目标检测
results = model("https://i0.hdslb.com/bfs/archive/a74136e5a14f5a5a0a9721dbf8d9f400ade61f05.jpg", show=True, save=True, project="runs/detect")