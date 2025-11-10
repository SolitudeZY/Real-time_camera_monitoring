#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Loader Module
===================

This module contains functions for loading pre-trained models
for object detection and segmentation.

Author: Assistant
Date: 2025-11-10
"""

import torch


class ModelLoader:
    """
    A class for loading pre-trained models.
    """
    
    def __init__(self, model_path):
        """
        Initialize the ModelLoader with a model path.
        
        Args:
            model_path (str): Path to the pre-trained model file
        """
        self.model_path = model_path
        self.model = None
    
    def load_model(self):
        """
        Load the pre-trained model.
        
        Returns:
            object: Loaded model object
        """
        try:
            # Try to load as a YOLOv8 model
            # This assumes ultralytics library is installed
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
            return self.model
        except ImportError:
            print("Warning: ultralytics library not found. Trying to load as PyTorch model.")
        except Exception as e:
            print(f"Warning: Could not load as YOLOv8 model: {e}. Trying as PyTorch model.")
        
        try:
            # Try to load as a generic PyTorch model
            self.model = torch.load(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
            return self.model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def get_model_info(self):
        """
        Get information about the loaded model.
        
        Returns:
            dict: Dictionary containing model information
        """
        if self.model is None:
            return {"error": "No model loaded"}
        
        info = {
            "model_path": self.model_path,
            "model_type": type(self.model).__name__
        }
        
        # Try to get additional information if available
        if hasattr(self.model, 'names'):
            info["class_names"] = self.model.names
        
        if hasattr(self.model, 'nc'):
            info["num_classes"] = self.model.nc
        
        return info