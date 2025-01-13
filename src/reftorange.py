import argparse
from functools import partial

import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results
import torch

from ouster.sdk.client import ChanField, LidarScan, ScanSource, destagger, FieldClass
from ouster.sdk import open_source
from ouster.sdk.client._utils import AutoExposure, BeamUniformityCorrector
from kumar import resize_to_half_width
from ouster.sdk.viz import SimpleViz


class ScanIterator:
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
class ScanIterator:
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

    def __init__(self, scans: ScanSource, use_opencv: bool = True):
        self._metadata = scans.metadata
        self._use_opencv = use_opencv  # Initialize the attribute

        # Load YOLO pretrained model
        self.model_yolo_nir = YOLO("yolov8x-seg.pt").to(device=self.DEVICE)
        self.model_yolo_ref = YOLO("yolov8x-seg.pt").to(device=self.DEVICE)

        # Define classes to output results for
        self.name_to_class = {value: key for key, value in self.model_yolo_ref.names.items()}
        self.classes_to_detect = [
            self.name_to_class['person'],
            self.name_to_class['car'],
            self.name_to_class['traffic light'],
            self.name_to_class['bus']
        ]

        # Post-process REFLECTIVITY channel
        self.paired_list = [
            #[ChanField.NEAR_IR, AutoExposure(), BeamUniformityCorrector(), self.model_yolo_nir],
            [ChanField.REFLECTIVITY, AutoExposure(), BeamUniformityCorrector(), self.model_yolo_ref]
        ]

        # Map the self._update function to the scans iterator
        self._scans = map(partial(self._update), scans)

    # Return the scans iterator when instantiating the class
    def __iter__(self):
        return self._scans

    def _update(self, scan: LidarScan) -> LidarScan:
        resized_width = 1024
        original_width = 2048
        resized_height = 128
        stacked_result_rgb = np.empty((resized_width * len(self.paired_list), resized_width, 3), np.uint8)

        for i, (field, ae, buc, model) in enumerate(self.paired_list):
        # Destagger and resize to downsampled dimensions
            img = destagger(self._metadata, scan.field(field)).astype(np.float32)
            img = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        # Apply uniformity and exposure corrections
            ae(img)
            buc(img, update_state=True)

        # Convert to RGB for YOLO inference
            img_rgb = np.repeat(np.uint8(np.clip(np.rint(img * 255), 0, 255))[..., np.newaxis], 3, axis=-1)

        # Run YOLO inference
            results: Results = next(
             model.track(
                   [img_rgb],
                    stream=True,
                    persist=True,
                     conf=0.1,
                      imgsz=[img.shape[0], img.shape[1]],
                      classes=self.classes_to_detect
                )
            ).cpu()

        # Scale bounding boxes back to original resolution
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                boxes_xyxy = boxes.xyxy.clone()
                boxes_xyxy[:, [0, 2]] *= original_width / resized_width  # Scale x-coordinates
                print(f"Scaled Bounding Boxes:\n{boxes_xyxy}")

        # Upsample masks to original resolution
            masks = results[0].masks
            if masks is not None:
                mask_tensor = masks.data.unsqueeze(1)  # Add channel dimension
                upsampled_mask_tensor = torch.nn.functional.interpolate(
                    mask_tensor,
                    size=(resized_height, original_width),  # Target size (H, W)
                    mode="nearest"
                )
                upsampled_masks = upsampled_mask_tensor.squeeze(1).cpu().numpy()  # Convert to NumPy
                print(f"Upsampled Mask Shape: {upsampled_masks.shape}")

        # Reproject results on the original LiDAR image
            original_img_rgb = cv2.cvtColor((cv2.resize(img, (original_width, resized_height)) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

            if masks is not None:
                for j in range(upsampled_masks.shape[0]):
                    mask = upsampled_masks[j]
                    colored_mask = np.zeros_like(original_img_rgb, dtype=np.uint8)
                    colored_mask[mask > 0] = (0, 255, 0)  # Green color for masks
                    alpha = 0.5
                    original_img_rgb = cv2.addWeighted(colored_mask, alpha, original_img_rgb, 1 - alpha, 0)

            if boxes is not None and len(boxes) > 0:
                for box in boxes_xyxy:
                    x1, y1, x2, y2 = map(int, box.tolist())
                    cv2.rectangle(original_img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue bounding box
            scan.add_field(f"YOLO_{field}", destagger(self._metadata, original_img_rgb, inverse=True))
        # Display the reprojected image
            if self._use_opencv:
                
                cv2.imshow("Reprojected YOLO Results", original_img_rgb)
                
            

                cv2.waitKey(1)

        return scan



if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(prog='sdk yolo demo',
                                     description='Runs a minimal demo of YOLO post-processing with OpenCV visualization')
    parser.add_argument('source', type=str, help='Sensor hostname or path to a sensor PCAP or OSF file')
    args = parser.parse_args()

    # Open source and process scans
    scans = ScanIterator(open_source(args.source, sensor_idx=0, cycle=True))
    for i, scan in enumerate(scans):
        if i > 20:  # Break after processing 20 frames
            break

    # Example for displaying results with SimpleViz
    scans = ScanIterator(open_source(args.source, sensor_idx=0, cycle=True), use_opencv=False)
    SimpleViz(scans._metadata, rate=0).run(scans)
