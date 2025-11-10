#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizer Module
=================

This module contains functions for visualizing results,
including 2D annotations and 3D point cloud visualization.

Author: Assistant
Date: 2025-11-10
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Visualizer:
    """
    A class for visualizing results of the 3D object measurement system.
    """
    
    def __init__(self):
        """
        Initialize the Visualizer.
        """
        pass
    
    def draw_bounding_boxes(self, image, boxes, class_ids=None, confidences=None, class_names=None):
        """
        Draw bounding boxes on an image.
        
        Args:
            image (np.ndarray): Input image
            boxes (list or np.ndarray): List of bounding boxes [x1, y1, x2, y2]
            class_ids (list, optional): List of class IDs for each box
            confidences (list, optional): List of confidence scores for each box
            class_names (dict, optional): Dictionary mapping class IDs to names
            
        Returns:
            np.ndarray: Image with bounding boxes drawn
        """
        annotated_image = image.copy()
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            
            # Draw rectangle
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare label
            label_parts = []
            
            if class_ids is not None and i < len(class_ids):
                class_id = class_ids[i]
                if class_names and class_id in class_names:
                    label_parts.append(class_names[class_id])
                else:
                    label_parts.append(f"Class {class_id}")
            
            if confidences is not None and i < len(confidences):
                label_parts.append(f"{confidences[i]:.2f}")
            
            label = " ".join(label_parts)
            
            # Put label text
            if label:
                cv2.putText(annotated_image, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return annotated_image
    
    def draw_masks(self, image, masks, color=(0, 0, 255), alpha=0.3):
        """
        Draw segmentation masks on an image.
        
        Args:
            image (np.ndarray): Input image
            masks (list or np.ndarray): List of masks
            color (tuple): Color for the masks (B, G, R)
            alpha (float): Transparency factor
            
        Returns:
            np.ndarray: Image with masks drawn
        """
        annotated_image = image.copy()
        
        for mask in masks:
            # Resize mask to match image size if needed
            if mask.shape[:2] != image.shape[:2]:
                mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
            else:
                mask_resized = mask
            
            # Create colored mask
            mask_colored = np.zeros_like(image)
            mask_colored[mask_resized > 0.5] = color
            
            # Blend with original image
            annotated_image = cv2.addWeighted(annotated_image, 1, mask_colored, alpha, 0)
        
        return annotated_image
    
    def plot_point_cloud(self, point_cloud, title="3D Point Cloud"):
        """
        Plot a 3D point cloud using matplotlib.
        
        Args:
            point_cloud (np.ndarray): Array of 3D points (N x 3)
            title (str): Title for the plot
        """
        if point_cloud.shape[1] != 3:
            print("Error: Point cloud must have 3 columns (x, y, z)")
            return
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Scatter plot
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], 
                  c=point_cloud[:, 2], cmap='viridis', s=50)
        
        # Set labels
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')
        ax.set_title(title)
        
        # Show plot
        plt.show()
    
    def save_point_cloud_plot(self, point_cloud, save_path, title="3D Point Cloud"):
        """
        Save a 3D point cloud plot to a file.
        
        Args:
            point_cloud (np.ndarray): Array of 3D points (N x 3)
            save_path (str): Path to save the plot image
            title (str): Title for the plot
        """
        if point_cloud.shape[1] != 3:
            print("Error: Point cloud must have 3 columns (x, y, z)")
            return
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Scatter plot
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], 
                  c=point_cloud[:, 2], cmap='viridis', s=50)
        
        # Set labels
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')
        ax.set_title(title)
        
        # Save plot
        plt.savefig(save_path)
        plt.close()