from flask import Flask, render_template, jsonify, request, Response
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time

app = Flask(__name__)

# 全局变量
model = None
camera = None
camera_lock = threading.Lock()
is_detecting = False

def init_model():
    """初始化模型"""
    global model
    try:
        model = YOLO("best.pt")
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        model = None

def generate_frames():
    """生成视频帧"""
    global camera, is_detecting
    
    while True:
        with camera_lock:
            if camera is None or not camera.isOpened():
                break
                
            success, frame = camera.read()
            if not success:
                break
                
            if is_detecting and model is not None:
                try:
                    # 使用模型进行预测
                    results = model.predict(
                        source=frame,
                        conf=0.5,
                        iou=0.5,
                        show=False,
                        verbose=False
                    )
                    
                    # 获取预测结果
                    result = results[0]
                    
                    # 绘制检测结果
                    annotated_frame = result.plot()
                except Exception as e:
                    print(f"检测出错: {e}")
                    annotated_frame = frame
            else:
                annotated_frame = frame
                
        # 编码帧
        try:
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"帧编码出错: {e}")
            continue

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """视频流"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """启动摄像头"""
    global camera, is_detecting
    
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                return jsonify({'status': 'error', 'message': '无法打开摄像头'})
                
    return jsonify({'status': 'success', 'message': '摄像头已启动'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """停止摄像头"""
    global camera, is_detecting
    
    with camera_lock:
        if camera is not None and camera.isOpened():
            camera.release()
            camera = None
            
    return jsonify({'status': 'success', 'message': '摄像头已停止'})

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    """切换检测状态"""
    global is_detecting
    
    is_detecting = not is_detecting
    status = "开启" if is_detecting else "关闭"
    return jsonify({'status': 'success', 'message': f'检测已{status}', 'detecting': is_detecting})

@app.route('/status')
def status():
    """获取状态"""
    global camera, is_detecting
    
    with camera_lock:
        camera_status = camera is not None and camera.isOpened()
        
    return jsonify({
        'camera_status': camera_status,
        'detecting': is_detecting,
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    # 初始化模型
    init_model()
    
    # 启动Flask应用
    app.run(debug=True, host='0.0.0.0', port=5000)