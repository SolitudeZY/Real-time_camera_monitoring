#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dimension Calculator Module
===========================

This module contains functions for calculating object dimensions
based on computer vision techniques and reference scaling.

Author: ZY
Date: 2025-11-10
"""

import cv2
import numpy as np
import csv


class DimensionCalculator:
    """
    A class for calculating dimensions of objects in images.
    """
    
    def __init__(self):
        """
        Initialize the DimensionCalculator.
        """
        pass
    
    def calculate_scale_factor(self, reference_image, known_width, known_height):
        """
        Calculate the scale factor (cm/pixel) based on a reference object.
        
        Args:
            reference_image (np.ndarray): Image of the reference object
            known_width (float): Known width of the reference object in cm
            known_height (float): Known height of the reference object in cm
            
        Returns:
            float: Scale factor in cm/pixel
        """
        # For simplicity, we'll assume the reference object is a rectangle
        # and we can detect its edges
        gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("Warning: No contours found in reference image")
            return None
        
        # Find the largest contour (assumed to be the reference object)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate scale factors for width and height
        width_scale = known_width / w if w > 0 else 0
        height_scale = known_height / h if h > 0 else 0
        
        # Return average scale factor
        scale_factor = (width_scale + height_scale) / 2
        return scale_factor
    
    def calculate_dimensions(self, bounding_box, scale_factor=None):
        """
        Calculate the dimensions of an object based on its bounding box.
        
        Args:
            bounding_box (list or np.ndarray): Bounding box coordinates [x1, y1, x2, y2]
            scale_factor (float, optional): Scale factor in cm/pixel
            
        Returns:
            tuple: Width and height of the object (in pixels or cm if scale_factor provided)
        """
        x1, y1, x2, y2 = bounding_box[:4]
        
        # Calculate width and height in pixels
        width_px = x2 - x1
        height_px = y2 - y1
        
        # If scale factor is provided, convert to cm
        if scale_factor is not None:
            width = width_px * scale_factor
            height = height_px * scale_factor
        else:
            # Return pixel dimensions
            width = width_px
            height = height_px
        
        return width, height
    
    def generate_point_cloud(self, object_measurements):
        """
        Generate a simplified 3D point cloud representation of detected objects.
        
        Args:
            object_measurements (list): List of dictionaries containing object measurements
            
        Returns:
            np.ndarray: Array of 3D points representing objects
        """
        # For this simplified implementation, we'll create a basic point cloud
        # with points at the corners of each bounding box
        points = []
        
        for obj in object_measurements:
            x1, y1, x2, y2 = obj['x1'], obj['y1'], obj['x2'], obj['y2']
            width, height = obj['width'], obj['height']
            
            # Add points at the corners of the bounding box
            # For a more realistic 3D reconstruction, depth information would be needed
            points.extend([
                [x1, y1, 0],  # Bottom-left corner
                [x2, y1, 0],  # Bottom-right corner
                [x2, y2, 0],  # Top-right corner
                [x1, y2, 0],  # Top-left corner
                [x1 + width/2, y1 + height/2, 10]  # Center point with arbitrary height
            ])
        
        return np.array(points)
    
    def save_measurements_to_csv(self, object_measurements, csv_path):
        """
        Save object measurements to a CSV file.
        
        Args:
            object_measurements (list): List of dictionaries containing object measurements
            csv_path (str or Path): Path to save the CSV file
        """
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['class_id', 'confidence', 'x1', 'y1', 'x2', 'y2', 'width', 'height']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for obj in object_measurements:
                writer.writerow(obj)