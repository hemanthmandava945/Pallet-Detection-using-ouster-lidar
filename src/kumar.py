import cv2
import numpy as np


# def correct_lidar_image_distortion(input_image, vertical_fov=45, horizontal_fov=360, interpolation=cv2.INTER_LINEAR):
#     """
#     Corrects distortion in LiDAR images based on angular field-of-view adjustments.
#     """
#     if len(input_image.shape) == 3:  # Check if the image has 3 channels (color)
#         input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

#     h, w = input_image.shape

#     # Initialize the remap grids
#     map_x = np.zeros((h, w), dtype=np.float32)
#     map_y = np.zeros((h, w), dtype=np.float32)

#     # Calculate angular resolution
#     vertical_angle_per_pixel = vertical_fov / h  # Vertical angular resolution
#     horizontal_angle_per_pixel = horizontal_fov / w  # Horizontal angular resolution

#     # Use horizontal angular resolution for square-like pixels
#     new_vertical_angle_per_pixel = horizontal_angle_per_pixel

#     # Populate remapping grids based on angular distribution
#     for y in range(h):
#         for x in range(w):
#             # Calculate the original angles for each pixel
#             theta = (x * horizontal_angle_per_pixel) - (horizontal_fov / 2)
#             phi = (y * vertical_angle_per_pixel) - (vertical_fov / 2)

#             # Map to corrected positions
#             corrected_x = ((theta + (horizontal_fov / 2)) / horizontal_fov) * w
#             corrected_y = ((phi + (vertical_fov / 2)) / vertical_fov) * h

#             # Fill the remapping arrays
#             map_x[y, x] = corrected_x
#             map_y[y, x] = corrected_y

#     # Apply the remapping to correct the distortion
#     corrected_image = cv2.remap(input_image, map_x, map_y, interpolation)
#     return corrected_image


def resize_to_half_width(input_image):
    """
    Reduces the width of the image by eliminating every 2nd column (down-sample horizontally by 2).
    This resizes an image from 2048 to 1024 width while maintaining height.

    :param input_image: Input image to be resized
    :return: Image resized to 1024x(height)
    """
    # Check the image dimensions
    h, w = input_image.shape[:2]
    if w != 2048:
        raise ValueError(f"Expected image width of 2048, but got {w}")

    # Downsample by selecting every alternate column
    resized_image = input_image[:, ::2]
    return resized_image

