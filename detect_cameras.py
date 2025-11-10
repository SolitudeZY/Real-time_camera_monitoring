import cv2
import os

def detect_cameras(max_cameras=10):
    """
    检测可用的摄像头设备
    :param max_cameras: 最大检测摄像头数量
    :return: 可用摄像头ID列表
    """
    available_cameras = []
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(i)
                print(f"摄像头 {i} 可用")
            cap.release()
    
    return available_cameras

def main():
    print("正在检测可用的摄像头设备...")
    cameras = detect_cameras()
    
    if cameras:
        print(f"\n检测到 {len(cameras)} 个可用摄像头:")
        for cam_id in cameras:
            print(f"  - 摄像头 {cam_id}")
        print("\n你可以使用以下命令运行摄像头检测程序:")
        print(f"python camera_detector.py --camera {cameras[0]}")
    else:
        print("未检测到可用的摄像头设备")

if __name__ == "__main__":
    main()