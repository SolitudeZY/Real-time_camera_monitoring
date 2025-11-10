#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Point Cloud Viewer
=================

This script loads and visualizes the saved point cloud data.

Author: Assistant
Date: 2025-11-10
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from pathlib import Path


def view_point_cloud(file_path):
    """
    Load and visualize point cloud data.
    
    Args:
        file_path (str or Path): Path to the .npy file containing point cloud data
    """
    try:
        # Load point cloud data
        point_cloud = np.load(file_path)
        
        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points
        xs = point_cloud[:, 0]
        ys = point_cloud[:, 1]
        zs = point_cloud[:, 2]
        
        ax.scatter(xs, ys, zs, c='r', marker='o')
        
        # Set labels
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_zlabel('Z (arbitrary units)')
        ax.set_title('3D Point Cloud Visualization')
        
        # Show plot
        plt.show()
        
        print(f"Point cloud loaded successfully from {file_path}")
        print(f"Shape of point cloud data: {point_cloud.shape}")
        print("Note: Z values are arbitrary in this simplified implementation.")
        
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        sys.exit(1)


def main():
    # Default path to point cloud file
    default_path = Path(__file__).parent / 'outputs' / 'point_cloud.npy'
    
    # Check if a file path was provided as command line argument
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
    else:
        file_path = default_path
    
    # Check if file exists
    if not file_path.exists():
        print(f"Error: Point cloud file not found at {file_path}")
        print("Please run the main.py script first to generate point cloud data.")
        sys.exit(1)
    
    # View point cloud
    view_point_cloud(file_path)


if __name__ == "__main__":
    main()