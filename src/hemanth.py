import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
# Load the image
image_path = '/home/mandava/processed_images/Linaer_image.jpg' # Replace with your image path
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
equalized_image = cv2.equalizeHist(img)
cv2.imwrite('/home/mandava/processed_images/equalise_LiDAR_image.jpg', equalized_image)
# Resize the image if necessary (optional)
# img = cv2.resize(img, (desired_width, desired_height))

# Check pixel range and type
# print("Min pixel value:", np.min(img))
# print("Max pixel value:", np.max(img))
# print("Pixel data type:", img.dtype)

# if img is None:
#     print("Error: Unable to load image")
# else:
#     # Get the shape of the image
#     height, width, channels = img.shape
#     print(f"Height: {height}, Width: {width}, Channels: {channels}")

# # Plot histogram of pixel values
# plt.hist(img.ravel(), bins=256, color='gray')
# plt.title('Histogram of Reflectivity Image Pixel Values')
# plt.xlabel('Pixel Value')
# plt.ylabel('Frequency')
# plt.savefig('/home/mandava/histogram2.png')
# plt.close()


