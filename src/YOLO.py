import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

# Load the YOLOv8 segmentation model
model = YOLO('yolov8x-seg.pt')

# Load your image
image_path = '/home/mandava/ir_images/resized/resized_ir_5.png'
image = cv2.imread(image_path)
original_shape = image.shape  # (H, W, C)
print(f"Original Image Shape: {original_shape}")

# Downsample the image by taking every alternate column
downsampled_image = image[:, ::2]
downsampled_shape = downsampled_image.shape
print(f"Downsampled Image Shape: {downsampled_shape}")

# Run YOLO inference
results = model(downsampled_image)

# Extract bounding boxes and masks
boxes = results[0].boxes
masks = results[0].masks
print(len(boxes))
print(masks.data)


# Upsample bounding boxes
if boxes is not None and len(boxes) > 0:
    boxes_xyxy = boxes.xyxy.clone()
    boxes_xyxy[:, [0, 2]] *= 2  # Scale x-coordinates back to original width

# Upsample segmentation masks
if masks is not None:
    mask_tensor = masks.data.unsqueeze(1)  # [N, 1, H, W_down]
    print(mask_tensor.shape)
    upsampled_mask_tensor = F.interpolate(
        mask_tensor,
        size=(original_shape[0], original_shape[1]),  # (H, W)
        mode='nearest'
    )
    upsampled_masks = upsampled_mask_tensor.squeeze(1).cpu().numpy()  # Convert to numpy
    print(upsampled_masks.shape[0])
# Overlay masks on the original image
overlay_image = image.copy()
color = (0, 255, 0)  # Green color for masks

for i in range(upsampled_masks.shape[0]):
    mask = upsampled_masks[i]
    colored_mask = np.zeros_like(overlay_image, dtype=np.uint8)
    colored_mask[mask > 0] = color
    alpha = 0.5
    overlay_image = cv2.addWeighted(colored_mask, alpha, overlay_image, 1 - alpha, 0)

# Draw bounding boxes
if boxes is not None and len(boxes) > 0:
    for box in boxes_xyxy:
        
        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(overlay_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue boxes
        print(overlay_image.shape)

# Display the overlaid image
cv2.imshow("Overlay", overlay_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


    


    # Apply rescaled masks to the original image
    # for i, mask in enumerate(rescaled_masks):
    #     mask_binary = (mask > 0.5).astype(np.uint8)  # Binarize the mask
    #     overlay = cv2.addWeighted(image, 0.8, (mask_binary * 255).astype(np.uint8), 0.2, 0)
    #     cv2.imshow(f"Mask {i}", overlay)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()


# new_boxes = torch.tensor([[100, 100, 200, 200, 0.9, 0]])
# results[0].update(boxes=new_boxes)
# annotated_image = results[0].plot()
# cv2.imshow("Updated YOLO Results", annotated_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# for result in results:
#      print(result.boxes)  # Print detection boxes
     








# Print results


# Annotate and display the results
# annotated_frame = results[0].plot()  # Annotated frame with detections
# cv2.imshow("YOLOv8 Detection", annotated_frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Save the output if needed
# output_path = '/home/mandava/processed_images/16.jpg'
# cv2.imwrite(output_path, annotated_frame)

#using python sdk and yolo

# import argparse
# from functools import partial

# import numpy as np
# import cv2
# from ultralytics import YOLO
# from ultralytics.engine.results import Results
# import torch

# from ouster.sdk import open_source

# # Example of using ScanBatcher instead of ScanIterator
# from ouster.sdk.client import LidarScan


# from ouster.sdk.client import ChanField, ScanSource, destagger

# from ouster.sdk.client._utils import AutoExposure, BeamUniformityCorrector
# from ouster.sdk.viz import SimpleViz

# if __name__ == '__main__':
#     # Parse the command arguments
#     parser = argparse.ArgumentParser(prog='sdk yolo demo',
#                                      description='Run a minimal demo of YOLO post-processing')
#     parser.add_argument('source', type=str, help='/path/to/your/pcap/file')
#     args = parser.parse_args()

#     # Open the sensor data source
#     scan_source = open_source(args.source, sensor_idx=0, cycle=True)

#     # Initialize ScanBatcher with the loaded scan source
#     for i, scan in enumerate(scan_source):
#         print(f"Frame {i}: {type(scan)}")  # Print type of scan
#         try:
#             destaggered = destagger(scan, scan_source.metadata)
#             print(f"Frame {i} destaggered shape: {destaggered.shape}")
#         except Exception as e:
#             print(f"Error destaggering Frame {i}: {e}")
#         if i > 20:  # Limit to 20 frames
#             break


#     # Display with SimpleViz
#     meta = scan_source.metadata
#     SimpleViz(meta, rate=0).run(scan_source)

# if __name__ == '__main__':
#     # parse the command arguments
#     parser = argparse.ArgumentParser(prog='sdk yolo demo',
#                                      description='Runs a minimal demo of yolo post-processing')
#     parser.add_argument('source', type=str, help='Sensor hostname or path to a sensor PCAP or OSF file')
#     args = parser.parse_args()

#     # Example for displaying results with opencv
#     scans = ScanIterator(open_source(args.source, sensor_idx=0, cycle=True), use_opencv=True)
#     for i, scan in enumerate(scans):
#         if i > 20:  # break after N frames
#             break

#     # Example for displaying results with SimpleViz
#     scans = open_source(args.source, sensor_idx=0, cycle=True)
#     meta = scans.metadata
#     scans = ScanIterator(scans, use_opencv=False)
#     SimpleViz(meta, rate=0).run(scans)