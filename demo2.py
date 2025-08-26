from ultralytics import YOLO
import cv2
import time


def main():
    # 加载训练好的模型
    model = YOLO("best.pt")  # 替换为您的模型路径

    # 打开摄像头
    cap = cv2.VideoCapture(0)  # 0 表示默认摄像头

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 设置窗口名称
    window_name = "YOLO 实时监测"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 设置帧率计算
    prev_time = 0
    fps = 0

    try:
        while True:
            # 读取摄像头帧
            ret, frame = cap.read()
            if not ret:
                print("无法获取帧")
                break

            # 使用模型进行预测
            results = model.predict(
                source=frame,
                conf=0.5,  # 置信度阈值
                iou=0.5,  # IOU阈值
                show=False,  # 不在内部显示结果
                verbose=False
            )

            # 获取预测结果
            result = results[0]

            # 绘制检测结果
            annotated_frame = result.plot()

            # 计算并显示FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
            prev_time = current_time

            # 在帧上显示FPS
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 显示结果
            cv2.imshow(window_name, annotated_frame)

            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()