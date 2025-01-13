import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import exposure
import os

def converting_images_to_gray(file_path):
    
    image_paths = [os.path.join(file_path, f"ir_frame_{i + 1}.png") for i in range(150)]
    grayscale_images = []

# Print each file path
    for file_path in image_paths:
        if os.path.isfile(file_path):
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Failed to load image: {file_path}")
            else:
                print(f"Grayscale image loaded successfully: {file_path}")
                grayscale_images.append((file_path, img))
        else:
            print(f"File not found: {file_path}")

    return grayscale_images