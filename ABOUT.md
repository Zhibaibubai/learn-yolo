## Web部署

本项目将YOLOv8目标检测模型部署到Web端，使用Flask框架提供Web服务。

### 项目结构

```
├── demo2.py          # 原始摄像头检测代码
├── demo3.py          # Flask后端代码
├── requirements.txt  # 项目依赖
├── best.pt           # YOLOv8模型文件
├── templates/
│   └── index.html    # 前端页面
└── README.md         # 项目说明文档
```

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行项目

```bash
python demo3.py
```

运行后，访问 `http://localhost:5000` 查看Web界面。

### 功能说明

1. **启动摄像头**：点击"启动摄像头"按钮打开摄像头
2. **停止摄像头**：点击"停止摄像头"按钮关闭摄像头
3. **目标检测**：点击"开启检测"按钮启用YOLOv8目标检测
4. **实时视频流**：在页面上显示实时摄像头画面

### 使用说明

1. 确保已训练好YOLOv8模型并保存为`best.pt`文件
2. 运行`demo3.py`启动Flask服务器
3. 在浏览器中访问`http://localhost:5000`
4. 点击"启动摄像头"按钮开始视频流
5. 点击"开启检测"按钮启用目标检测功能

### 注意事项

- 需要确保计算机有摄像头设备
- 模型文件`best.pt`必须在项目根目录下
- 首次运行时会自动下载YOLOv8相关依赖