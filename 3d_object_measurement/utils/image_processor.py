#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Processor Module
======================

This module contains functions for image processing tasks such as
loading, preprocessing, and basic manipulations.

Author: Assistant
Date: 2025-11-10
"""

import cv2
import numpy as np


class ImageProcessor:
    """
    A class for processing images for the 3D object measurement system.
    """
    
    def __init__(self):
        """
        Initialize the ImageProcessor.
        """
        pass
    
    def preprocess_image(self, image):
        """
        Preprocess an image for object detection.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Preprocessed image
        """
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        return rgb_image
    
    def enhance_image(self, image):
        """
        Apply image enhancement techniques.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced_image
    
    def detect_edges(self, image, threshold1=50, threshold2=150):
        """
        Detect edges in an image using Canny edge detection.
        
        Args:
            image (np.ndarray): Input image
            threshold1 (int): First threshold for hysteresis procedure
            threshold2 (int): Second threshold for hysteresis procedure
            
        Returns:
            np.ndarray: Edge-detected image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1, threshold2)
        return edges
    
    def find_contours(self, image):
        """
        Find contours in an image.
        
        Args:
            image (np.ndarray): Input image (binary)
            
        Returns:
            list: List of contours
        """
        # Ensure image is binary
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Threshold if not already binary
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours