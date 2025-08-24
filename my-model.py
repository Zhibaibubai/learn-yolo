# 开始训练模型

from ultralytics import YOLO

# 加载预训练模型
model = YOLO("yolov8n.pt")

# 开始训练模型
model.train(
    data="data.yaml",   # 数据集配置文件路径
    epochs=1000,    # 训练轮次
    batch=16,  # 批次大小
    img=640,    # 输入图像大小
    device="cuda"   # 训练设备
)

print("训练完成")