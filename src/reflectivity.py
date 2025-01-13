import numpy as np
import cv2
import matplotlib.pyplot as plt
from ouster.sdk.client._utils import AutoExposure
from ultralytics import YOLO
import os



# Function to generate and save histograms
def generate_histogram(image, title, save_path):
    plt.hist(image.ravel(), bins=256, range=(0, 255), color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.savefig(save_path)
    plt.clf()


# Function to apply row-wise normalization
def row_wise_normalization(image):
    normalized_image = np.zeros_like(image, dtype=np.float32)
    row_means = []

    for i, row in enumerate(image):
        row_mean = np.mean(row)
        row_means.append(row_mean)
        normalized_image[i] = row - row_mean

    # Rescale to original intensity range (0–255)
    normalized_image = normalized_image - normalized_image.min()
    normalized_image = (normalized_image / normalized_image.max()) * 255

    return normalized_image.astype(np.uint8), row_means


# Main pipeline
if __name__ == "__main__":
    # File paths and setup
    reflectivity_images = [f"/home/mandava/rangeimages/extract{i+1}.jpg" for i in range(150)]  # Provide image paths
    base_output_dir = "/home/mandava/histogram_images"  # Base directory to save all results
    resized_hist_dir = os.path.join(base_output_dir, "resized_histograms")  # Subdirectory for resized histograms
    normalized_hist_dir = os.path.join(base_output_dir, "normalized_histograms")  # Subdirectory for normalized histograms
    inference_dir = os.path.join(base_output_dir, "inferences")  # Subdirectory for YOLO inferences

    # Create directories if they don't exist
    os.makedirs(resized_hist_dir, exist_ok=True)
    os.makedirs(normalized_hist_dir, exist_ok=True)
    os.makedirs(inference_dir, exist_ok=True)

    model = YOLO('yolov8x-seg.pt')  # Load YOLOv8 segmentation model
    auto_exposure = AutoExposure()

    for idx, img_path in enumerate(reflectivity_images):
        print(f"Processing image {idx + 1}/{len(reflectivity_images)}: {img_path}")

        # Step 1: Load the reflectivity image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to load image: {img_path}")
            continue

        # Step 2: Resize the image to 1024x324 using bicubic interpolation
        resized_image = cv2.resize(image, (1024, 324), interpolation=cv2.INTER_CUBIC)

        # Step 3: Generate and save histogram of resized image
        resized_hist_path = os.path.join(resized_hist_dir, f"hist_resized_{idx + 1}.png")
        cv2.imwrite(resized_hist_path, resized_image)
        #generate_histogram(resized_image, f"Resized Image Histogram (Image {idx + 1})", resized_hist_path)
        
        
        print(f"Saved resized IR image {idx + 1}.")

        # Step 4: Apply auto-exposure
        enhanced_image = resized_image.astype(np.float32)
        auto_exposure(enhanced_image)
        enhanced_image = np.clip(np.rint(enhanced_image * 255), 0, 255).astype(np.uint8)

        # Step 5: Apply row-wise normalization
        normalized_image, row_means = row_wise_normalization(enhanced_image)

        # Step 6: Generate and save histogram of normalized image
        normalized_hist_path = os.path.join(normalized_hist_dir, f"hist_normalized_{idx + 1}.png")
        cv2.imwrite(normalized_hist_path, normalized_image)
        #generate_histogram(normalized_image, f"Normalized Image Histogram (Image {idx + 1})", normalized_hist_path)
        

        # Step 7: YOLO inference
        input_image = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel BGR for YOLO
        results = model(input_image)

        # Step 8: Save inference result
        inference_path = os.path.join(inference_dir, f"inference_image_{idx + 1}.png")
        result_image = results[0].plot()  # Visualize predictions
        cv2.imwrite(inference_path, result_image)
        print(f"Saved inference result: {inference_path}")
    

    print("Processing complete!")