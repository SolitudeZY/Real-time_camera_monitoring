import cv2
import numpy as np
from ultralytics import YOLO
import time
import argparse

class CameraDetector:
    def __init__(self, model_path='yolov8n.pt', save_dir='results', resolution=(1280, 720)):
        """
        初始化摄像头检测器
        :param model_path: YOLO模型路径
        :param save_dir: 结果保存目录
        :param resolution: 摄像头分辨率 (width, height)
        """
        # 加载YOLO模型
        self.model = YOLO(model_path)
        
        # 摄像头相关参数
        self.cap = None
        self.running = False
        self.resolution = resolution
        
        # 性能统计
        self.frame_count = 0
        self.start_time = 0
        self.fps = 0
        
        # 保存相关
        self.save_dir = save_dir
        self.recording = False
        self.out = None
        
        # 目标统计
        self.detection_stats = {}
        
        # 创建保存目录
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def start_camera(self, camera_id=0):
        """
        启动摄像头
        :param camera_id: 摄像头ID
        """
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {camera_id}")
        
        # 设置分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        self.running = True
        self.frame_count = 0
        self.start_time = time.time()
        
        print(f"摄像头已启动，分辨率: {self.resolution[0]}x{self.resolution[1]}")
        print("按 'q' 退出，按 's' 保存当前帧，按 'r' 开始/停止录制")
    
    def process_frame(self, frame):
        """
        处理单帧图像
        :param frame: 输入图像帧
        :return: 处理后的图像帧
        """
        # 记录开始时间
        start_time = time.time()
        
        # 使用YOLO模型进行目标检测
        results = self.model(frame)
        
        # 获取检测结果
        result = results[0]
        
        # 绘制检测结果
        annotated_frame = result.plot()
        
        # 更新目标统计信息
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            class_names = result.names if hasattr(result, 'names') else {}
            
            # 重置当前帧统计
            frame_stats = {}
            
            # 统计各类别目标数量
            for box in boxes:
                class_id = int(box.cls)
                class_name = class_names.get(class_id, f"Class {class_id}")
                
                if class_name in frame_stats:
                    frame_stats[class_name] += 1
                else:
                    frame_stats[class_name] = 1
                
                # 更新总统计
                if class_name in self.detection_stats:
                    self.detection_stats[class_name] += 1
                else:
                    self.detection_stats[class_name] = 1
            
            # 在图像上显示检测到的目标统计
            y_offset = 110
            for class_name, count in frame_stats.items():
                cv2.putText(annotated_frame, f"{class_name}: {count}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30
        
        # 计算并显示FPS（使用滑动窗口计算瞬时FPS）
        self.frame_count += 1
        if self.frame_count % 10 == 0:  # 每10帧更新一次FPS
            elapsed_time = time.time() - self.start_time
            self.fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            self.start_time = time.time()
            self.frame_count = 0
        
        # 在图像上显示FPS和其他信息
        if hasattr(self, 'fps'):
            cv2.putText(annotated_frame, f"FPS: {self.fps:.2f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示推理时间
        inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
        cv2.putText(annotated_frame, f"Inference: {inference_time:.2f}ms", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return annotated_frame
    
    def save_frame(self, frame, filename=None):
        """
        保存当前帧
        :param frame: 要保存的图像帧
        :param filename: 文件名
        """
        import os
        if filename is None:
            timestamp = int(time.time())
            filename = f"frame_{timestamp}.jpg"
        
        filepath = os.path.join(self.save_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"帧已保存到: {filepath}")
    
    def start_recording(self, frame):
        """
        开始录制视频
        :param frame: 当前帧（用于获取视频尺寸）
        """
        if self.recording:
            return
        
        timestamp = int(time.time())
        filename = f"recording_{timestamp}.mp4"
        filepath = os.path.join(self.save_dir, filename)
        
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(filepath, fourcc, 20.0, (width, height))
        
        self.recording = True
        print(f"开始录制到: {filepath}")
    
    def stop_recording(self):
        """
        停止录制视频
        """
        if not self.recording:
            return
        
        self.recording = False
        if self.out:
            self.out.release()
            self.out = None
        
        print("录制已停止")
    
    def run(self):
        """
        运行主循环
        """
        if not self.running:
            raise RuntimeError("摄像头未启动")
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break
            
            # 处理帧
            processed_frame = self.process_frame(frame)
            
            # 如果正在录制，则写入视频
            if self.recording and self.out:
                self.out.write(processed_frame)
            
            # 显示结果
            cv2.imshow('YOLO Camera Detector', processed_frame)
            
            # 处理按键事件
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # 退出
                break
            elif key == ord('s'):  # 保存当前帧
                self.save_frame(processed_frame)
            elif key == ord('r'):  # 开始/停止录制
                if self.recording:
                    self.stop_recording()
                else:
                    self.start_recording(processed_frame)
        
        # 清理资源
        self.cleanup()
    
    def cleanup(self):
        """
        清理资源
        """
        self.running = False
        
        if self.recording:
            self.stop_recording()
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # 显示最终FPS
        elapsed_time = time.time() - self.start_time
        final_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        print(f"\n总帧数: {self.frame_count}")
        print(f"总时间: {elapsed_time:.2f} 秒")
        print(f"平均FPS: {final_fps:.2f}")
        
        # 显示目标检测统计
        print("\n目标检测统计:")
        for class_name, count in self.detection_stats.items():
            print(f"  {class_name}: {count}")

def main():
    parser = argparse.ArgumentParser(description='基于YOLO的实时摄像头监测系统')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLO模型路径')
    parser.add_argument('--camera', type=int, default=0, help='摄像头ID')
    parser.add_argument('--width', type=int, default=1280, help='摄像头宽度')
    parser.add_argument('--height', type=int, default=720, help='摄像头高度')
    args = parser.parse_args()
    
    detector = CameraDetector(model_path=args.model, resolution=(args.width, args.height))
    
    try:
        detector.start_camera(camera_id=args.camera)
        detector.run()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        detector.cleanup()

if __name__ == '__main__':
    main()