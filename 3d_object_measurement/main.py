#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Object Measurement System
=============================

This script implements a 3D object measurement system using computer vision techniques.
It can capture images from a camera or load images from files, detect objects,
perform 3D reconstruction, and measure object dimensions.

Author: Assistant
Date: 2025-11-10
"""

import cv2
import numpy as np
import argparse
import os
import sys
from pathlib import Path

# Add the project root to the path to import modules
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    from utils.image_processor import ImageProcessor
    from utils.dimension_calculator import DimensionCalculator
    from utils.model_loader import ModelLoader
    from utils.visualizer import Visualizer
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all required modules are implemented.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='3D Object Measurement System')
    parser.add_argument('--source', type=str, default='0', 
                        help='Camera ID (integer) or path to image file')
    parser.add_argument('--model', type=str, default='yolov8n-seg.pt', 
                        help='Path to the pre-trained model file')
    parser.add_argument('--reference', type=str, 
                        help='Path to reference object image for calibration')
    parser.add_argument('--ref-width', type=float, default=0.0,
                        help='Known width of the reference object in cm')
    parser.add_argument('--ref-height', type=float, default=0.0,
                        help='Known height of the reference object in cm')
    
    args = parser.parse_args()
    
    # Initialize components
    try:
        model_loader = ModelLoader(args.model)
        model = model_loader.load_model()
        
        image_processor = ImageProcessor()
        dimension_calculator = DimensionCalculator()
        visualizer = Visualizer()
        
        print("3D Object Measurement System Initialized")
        print(f"Using model: {args.model}")
        
        # Determine if source is a camera or image file
        if args.source.isdigit():
            # Camera input
            cap = cv2.VideoCapture(int(args.source))
            if not cap.isOpened():
                print(f"Error: Could not open camera {args.source}")
                return
            
            print(f"Camera {args.source} opened successfully")
            print("Press 'q' to quit, 's' to save current frame")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    break
                
                # Process frame
                processed_frame = process_frame(
                    frame, model, image_processor, dimension_calculator, 
                    visualizer, args
                )
                
                # Display the result
                cv2.imshow('3D Object Measurement', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save frame
                    save_path = project_root / 'outputs' / f'frame_{int(cv2.getTickCount())}.jpg'
                    cv2.imwrite(str(save_path), processed_frame)
                    print(f"Frame saved to {save_path}")
            
            cap.release()
        else:
            # Image file input
            if not os.path.exists(args.source):
                print(f"Error: Image file {args.source} not found")
                return
            
            frame = cv2.imread(args.source)
            if frame is None:
                print(f"Error: Could not load image from {args.source}")
                return
            
            print(f"Processing image: {args.source}")
            
            # Process frame
            processed_frame = process_frame(
                frame, model, image_processor, dimension_calculator, 
                visualizer, args
            )
            
            # Display the result
            cv2.imshow('3D Object Measurement - Result', processed_frame)
            print("Press any key to close")
            cv2.waitKey(0)
        
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error during initialization: {e}")
        sys.exit(1)


def process_frame(frame, model, image_processor, dimension_calculator, visualizer, args):
    """
    Process a single frame: detect objects, calculate dimensions, and visualize results.
    """
    try:
        # Perform object detection and segmentation
        results = model(frame)
        
        # Extract masks and bounding boxes
        if hasattr(results[0], 'masks') and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes.data.cpu().numpy()
            class_ids = boxes[:, 5].astype(int) if boxes.shape[1] > 5 else []
            confidences = boxes[:, 4] if boxes.shape[1] > 4 else []
        else:
            print("Warning: No masks found in detection results. Using bounding boxes only.")
            masks = None
            boxes = results[0].boxes.data.cpu().numpy() if results[0].boxes is not None else np.array([])
            class_ids = boxes[:, 5].astype(int) if boxes.shape[1] > 5 else []
            confidences = boxes[:, 4] if boxes.shape[1] > 4 else []
        
        # Prepare for visualization
        annotated_frame = frame.copy()
        
        # Perform dimension calculation if reference is provided
        scale_factor = None
        if args.reference and args.ref_width > 0 and args.ref_height > 0:
            # Load reference image
            ref_image = cv2.imread(args.reference)
            if ref_image is not None:
                # Calculate scale factor based on reference object
                scale_factor = dimension_calculator.calculate_scale_factor(
                    ref_image, args.ref_width, args.ref_height
                )
                print(f"Scale factor calculated: {scale_factor} cm/pixel")
        
        # Process each detected object
        object_measurements = []
        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = map(int, box[:4])
            confidence = confidences[i] if len(confidences) > i else 0
            class_id = class_ids[i] if len(class_ids) > i else 0
            
            # Extract mask if available
            mask = masks[i] if masks is not None and len(masks) > i else None
            
            # Calculate dimensions
            width, height = dimension_calculator.calculate_dimensions(
                box, scale_factor
            )
            
            # Store measurements
            object_measurements.append({
                'class_id': class_id,
                'confidence': confidence,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'width': width, 'height': height
            })
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw mask if available
            if mask is not None:
                # Resize mask to match frame size
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask_colored = np.zeros_like(frame)
                mask_colored[mask_resized > 0.5] = (0, 0, 255)  # Red color for mask
                annotated_frame = cv2.addWeighted(annotated_frame, 1, mask_colored, 0.3, 0)
            
            # Display label and dimensions
            label = f"Class {class_id}: {confidence:.2f}"
            dim_label = f"{width:.2f}cm x {height:.2f}cm" if scale_factor else f"{x2-x1}px x {y2-y1}px"
            
            # Put text on frame
            cv2.putText(annotated_frame, label, (x1, y1 - 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(annotated_frame, dim_label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Generate 3D point cloud (simplified representation)
        if len(object_measurements) > 0:
            point_cloud = dimension_calculator.generate_point_cloud(object_measurements)
            # Save point cloud data
            output_path = project_root / 'outputs' / 'point_cloud.npy'
            np.save(output_path, point_cloud)
            print(f"Point cloud saved to {output_path}")
        
        # Save measurements to CSV
        if len(object_measurements) > 0:
            csv_path = project_root / 'outputs' / 'measurements.csv'
            dimension_calculator.save_measurements_to_csv(object_measurements, csv_path)
            print(f"Measurements saved to {csv_path}")
        
        return annotated_frame
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return frame


if __name__ == "__main__":
    main()