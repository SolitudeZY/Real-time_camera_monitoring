# 基于YOLO的实时摄像头监测系统

本项目是一个基于YOLO模型的实时摄像头监测系统，能够实时检测摄像头画面中的目标物体。

## 功能特点

- 实时摄像头画面捕获
- 基于YOLOv8的目标检测
- 边界框、类别标签和置信度分数的可视化显示
- 实时FPS和推理时间显示
- 目标检测统计信息显示
- 支持多种分辨率
- 可选的检测结果保存功能

## 环境依赖

请查看 `requirements.txt` 文件。

## 安装步骤

1. 克隆此仓库到本地：
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. 安装所需的依赖包：
   ```bash
   pip install -r requirements.txt
   ```

3. 下载YOLO模型文件：
   YOLOv8模型文件会自动从Ultralytics的最新版本下载，首次使用时会自动下载默认的`yolov8n.pt`模型。
   如果需要其他模型，可以从以下链接手动下载：
   - [YOLOv8 Detection Models](https://docs.ultralytics.com/models/yolov8/#performance)
   - [Ultralytics YOLOv8 HuggingFace](https://huggingface.co/Ultralytics/YOLOv8)

## 使用方法

### 自动检测摄像头设备

首先，可以使用我们提供的脚本自动检测可用的摄像头设备：

```bash
python detect_cameras.py
```

### 运行摄像头监测系统

运行以下命令启动摄像头监测系统：

```bash
python camera_detector.py [--model MODEL_PATH] [--camera CAMERA_ID] [--width WIDTH] [--height HEIGHT]
```

参数说明：
- `--model`: 指定YOLO模型文件路径（可选，默认使用yolov8n.pt）
- `--camera`: 指定摄像头ID（可选，默认使用0）
- `--width`: 指定摄像头宽度（可选，默认使用1280）
- `--height`: 指定摄像头高度（可选，默认使用720）

示例：
```bash
python camera_detector.py --model yolov8s.pt --camera 1 --width 640 --height 480
```

## 可选功能

- 按 's' 键保存当前帧的检测结果
- 按 'r' 键开始/停止录制检测结果视频
- 程序结束时显示检测到的目标统计信息