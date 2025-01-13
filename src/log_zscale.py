import os
import numpy as np
from matplotlib import pyplot as plt
from astropy.visualization import ZScaleInterval
import cv2
from ultralytics import YOLO

# Input and output directories
input_dir = "/home/mandava/ir_images/Raw_Images1"
output_dir = "/home/mandava/ir_images/Processed_Images15-12"
yolo_input_images = "/home/mandava/ir_images/Processed_Images15-12/"
yolo_out_images = "/home/mandava/ir_images/yolo_out_images"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(yolo_out_images, exist_ok=True)
model = YOLO('yolov8x-seg.pt')

# Functions for normalization
def log_normalization(ir_pixels):
    return np.log(ir_pixels)

def reverse_log_normalization(log_pixels):
    return np.exp(log_pixels)

def process_image(file_path, output_dir):
    # Load the .npy file
    ir_image = np.load(file_path)
    print(f"Processing {file_path}: Min = {np.min(ir_image)}, Max = {np.max(ir_image)}")
def yolo_inference(idx,image):
    input_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    results = model(image)
    inference_result = results[0].plot()  # Visualize predictions
    inference_path = os.path.join(yolo_out_images, f"inference_image_{idx + 1}.png")
    cv2.imwrite(inference_path, inference_result)
    print(f"Saved YOLO inference result for image {idx + 1}.")

    print("Processing complete!")


    # Log normalization
    log_image = log_normalization(ir_image)

def save_zscaled_image(ir_image, output_dir, file_name):
    """Normalize, resize, Z-scale, and save an image."""
    log_image = log_normalization(ir_image)

    # Resize to (1024, 324)
    resized_image = cv2.resize(log_image, (1024, 324), interpolation=cv2.INTER_CUBIC)

    # Z-scale normalization
    z = ZScaleInterval()
    z1, z2 = z.get_limits(resized_image)
    print(resized_image)
    z_scaled_image = np.clip((resized_image - z1) / (z2 - z1) * 255, 0, 255).astype(np.uint8)
    print(z_scaled_image)

    # Save as PNG
    save_path = os.path.join(output_dir, f"z_scaled_{file_name.replace('.npy', '.png')}")
    cv2.imwrite(save_path, z_scaled_image)
    print(f"Saved Z-scaled image: {save_path}")

    return z_scaled_image  # Return for further processing (e.g., YOLO)

def yolo_inference(idx, image, output_dir):
    """Run YOLO inference on an image and save results."""
    # Ensure the image is 3-channel BGR
    input_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Run YOLO inference
    results = model(input_image)

    # Save inference result
    inference_result = results[0].plot()  # Visualize predictions
    inference_path = os.path.join(output_dir, f"inference_image_{idx + 1}.png")
    cv2.imwrite(inference_path, inference_result)
    print(f"Saved YOLO inference result for image {idx + 1}.")

# Process all .npy files
for file_name in os.listdir(input_dir):
    if file_name.endswith(".npy"):
        file_path = os.path.join(input_dir, file_name)

        # Load .npy file
        ir_image = np.load(file_path)
        print(f"Processing {file_name}: Min = {np.min(ir_image)}, Max = {np.max(ir_image)}")

        # Process and save Z-scaled image
        processed_image = save_zscaled_image(ir_image, output_dir, file_name)

# YOLO inference on processed images
for idx, image_name in enumerate(os.listdir(yolo_input_images)):
    if image_name.endswith((".jpg", ".png")):
        image_path = os.path.join(yolo_input_images, image_name)

        # Read image for YOLO
        image_for_yolo = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Run inference
        yolo_inference(idx, image_for_yolo, yolo_out_images)

print("All images processed and saved.")

    # # Resize the image to (1024, 324)
    # resized_image = cv2.resize(log_image, (1024, 324), interpolation=cv2.INTER_CUBIC)
    


    # # Z-scale normalization
    # z = ZScaleInterval()
    # z1, z2 = z.get_limits(resized_image)
    # print(f"Z-scale limits: Min = {z1}, Max = {z2}")

    # # Save the Z-scaled image
    # plt.figure(figsize=(10, 6))
    # plt.imshow(resized_image, cmap='gray', vmin=z1, vmax=z2)
    # plt.colorbar(label="Intensity")
    # plt.title(f"Z-Scaled Image ({os.path.basename(file_path)})")
    # plt.xlabel("Width")
    # plt.ylabel("Height")
    
    # # Save to output directory
    # save_path = os.path.join(output_dir, f"z_scaled_{os.path.basename(file_path).replace('.npy', '.png')}")
    # plt.savefig(save_path)
    # plt.clf()
    # print(f"Saved Z-scaled image: {save_path}")



# Process all .npy files in the input directory
# for file_name in os.listdir(input_dir):
#     if file_name.endswith(".npy"):
#         file_path = os.path.join(input_dir, file_name)
#         process_image(file_path, output_dir)

# for idx,image_name in enumerate(os.listdir(yolo_input_images)):
#     if image_name.endswith((".jpg",".png")):
#         image_path = os.path.join(yolo_input_images,image_name)

#         #image_for_yolo = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#         b = yolo_inference(image_path,idx)


# print("All images processed and saved.")
