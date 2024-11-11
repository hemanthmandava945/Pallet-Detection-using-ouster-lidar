from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Load your image
image_path = '/home/mandava/processed_images/equalise_LiDAR_image.jpg'   # replace with your image path
image = cv2.imread(image_path)

# Run inference
results = model(image)

# Print results
print(results)

# Annotate and display the results
annotated_frame = results[0].plot()  # Annotated frame with detections
cv2.imshow("YOLOv8 Detection", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the output if needed
output_path = '/home/mandava/processed_images/yolo_LiDAR_image.jpg'
cv2.imwrite(output_path, annotated_frame)
