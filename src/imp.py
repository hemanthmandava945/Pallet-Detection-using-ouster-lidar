import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
# Load the 
image_path =  '/home/mandava/rangeimages/extract102.jpg'

img = cv2.imread(image_path)

if img is None:
    print("Image is not present")
    sys.exit()
#resize the image 
new_dimensions = (1024,324)
img_resized = cv2.resize(img,new_dimensions,interpolation = cv2.INTER_LINEAR)

output_path = '/home/mandava/processed_images/Linaer_image.jpg'  # Change to your desired output path
cv2.imwrite(output_path, img_resized)

print(f"Resized image saved to {output_path}")

