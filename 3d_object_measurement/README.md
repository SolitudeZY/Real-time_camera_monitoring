# 3D Object Measurement System

This project implements a 3D object measurement system using computer vision techniques. It can capture images from a camera or load images from files, detect objects, perform 3D reconstruction, and measure object dimensions.

## Features

- Capture images from a camera or load from files
- Object detection and segmentation using deep learning models (YOLOv8)
- 3D reconstruction based on monocular vision
- Dimension measurement with reference object calibration
- Visualization of results (2D annotations and 3D point cloud)
- Export measurements to CSV format

## Directory Structure

```
3d_object_measurement/
├── main.py              # Main program entry
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── utils/               # Utility functions
│   ├── image_processor.py
│   ├── dimension_calculator.py
│   ├── model_loader.py
│   └── visualizer.py
├── models/              # Pre-trained model weights
└── outputs/             # Output results and 3D models
```

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- Matplotlib
- PyTorch
- Ultralytics YOLOv8

## Installation

1. Clone the repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Basic usage with camera:

```bash
python main.py --source 0
```

### With a specific model:

```bash
python main.py --source 0 --model yolov8n-seg.pt
```

### With image file input:

```bash
python main.py --source path/to/image.jpg
```

### With reference object for calibration:

```bash
python main.py --source 0 --reference path/to/reference.jpg --ref-width 10.0 --ref-height 5.0
```

### Command Line Arguments

- `--source`: Camera ID (integer) or path to image file (default: 0)
- `--model`: Path to the pre-trained model file (default: yolov8n-seg.pt)
- `--reference`: Path to reference object image for calibration
- `--ref-width`: Known width of the reference object in cm
- `--ref-height`: Known height of the reference object in cm

## Implementation Details

### Image Processing

The `ImageProcessor` class handles basic image processing tasks including preprocessing, enhancement, edge detection, and contour finding.

### Dimension Calculation

The `DimensionCalculator` class performs dimension calculations based on bounding boxes and reference object scaling. It also generates simplified 3D point clouds and exports measurements to CSV.

### Model Loading

The `ModelLoader` class handles loading of pre-trained models, primarily supporting YOLOv8 models from Ultralytics.

### Visualization

The `Visualizer` class provides functions for drawing bounding boxes, segmentation masks, and 3D point cloud visualization.

### Viewing Point Cloud Data

The system generates point cloud data and saves it as a NumPy array in the `outputs/point_cloud.npy` file. To view this data:

1. Run the provided point cloud viewer script:

```bash
python view_point_cloud.py
```

2. Or load and examine the data manually in Python:

```python
import numpy as np
point_cloud = np.load('outputs/point_cloud.npy')
print(point_cloud.shape)
print(point_cloud)
```

## Notes

- This implementation requires either a depth camera or a reference object of known dimensions for accurate measurements.
- The 3D reconstruction is simplified and represents a basic point cloud. For more accurate 3D models, additional depth information would be needed.
- The Z-coordinates in the point cloud are arbitrary in this simplified implementation. A more realistic 3D reconstruction would require depth information from a depth camera or stereo vision.