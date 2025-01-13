from ouster.sdk import open_source
import ouster.sdk.viz as viz
import numpy as np

import open3d as o3d
from ouster.sdk import client
from ouster.sdk.client import ChanField, LidarScan

# Creating a point viz instance
# point_viz = viz.PointViz("Example Viz")#name on tab opened
# viz.add_default_controls(point_viz)
# img = viz.Image()
# img_array = np.full((10, 10), 0.5, dtype=np.float32) 
# img.set_image(img_array) 
# img.set_position(0,1,-0.5,0.5)
# point_viz.add(img)

# # ... add objects here

# # update internal objects buffers and run visualizer
# point_viz.update()
# point_viz.run()
# Initialize the point visualizer
point_viz = viz.PointViz("LiDAR Visualization")
viz.add_default_controls(point_viz)
pcap_path = "/home/mandava/Downloads/OS-0-128_v3.0.1_2048x10_20230216_173241-000.pcap"
metadata_path = "/home/mandava/Downloads/OS-0-128_v3.0.1_2048x10_20230216_173241.json"
from ouster.sdk import open_source
source = open_source(pcap_path, meta=[metadata_path])
metadata = source.metadata

source_iter = iter(source)
#ctr = 0  # Ensure the counter is initialized before using it
scan = next(source_iter)
img_aspect = (metadata.beam_altitude_angles[0] -
              metadata.beam_altitude_angles[-1]) /360
img_screen_height = 0.5  # [0..2]
img_screen_len = img_screen_height / img_aspect

ranges = scan.field(client.ChanField.RANGE)
ranges = client.destagger(metadata, ranges)
ranges = np.divide(ranges, np.amax(ranges), dtype=np.float32)

signal = scan.field(client.ChanField.REFLECTIVITY)
signal = client.destagger(metadata, signal)
signal = np.divide(signal, np.amax(signal), dtype=np.float32)

range_image = viz.Image()
range_image.set_image(ranges)
range_image.set_position(-img_screen_len / 2, img_screen_len / 2, 1 - img_screen_height, 1)
point_viz.add(range_image)
signal_img = viz.Image()
signal_img.set_image(signal)
img_aspect = (metadata.beam_altitude_angles[0] -
              metadata.beam_altitude_angles[-1]) / 360.0
img_screen_height = 0.4  # [0..2]
img_screen_len = img_screen_height / img_aspect
# bottom center position
signal_img.set_position(-img_screen_len / 2, img_screen_len / 2, -1,-1 + img_screen_height)
point_viz.add(signal_img)
cloud_scan = viz.Cloud(metadata)
cloud_scan.set_range(scan.field(client.ChanField.RANGE))
cloud_scan.set_key(ranges)
point_viz.add(cloud_scan)
point_viz.update()
point_viz.run()