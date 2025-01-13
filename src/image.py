import numpy as np
import open3d as o3d
from ouster.sdk import client
from ouster.sdk.client import ChanField, LidarScan
import matplotlib.pyplot as plt
from more_itertools import nth


pcap_path = "/home/mandava/Downloads/OS-0-128_v3.0.1_2048x10_20230216_173241-000.pcap"
metadata_path = "/home/mandava/Downloads/OS-0-128_v3.0.1_2048x10_20230216_173241.json"
from ouster.sdk import open_source
source = open_source(pcap_path, meta=[metadata_path])
metadata = source.metadata
print(metadata)
print(source)
# print("\nLiDAR Configuration:")
# print(f"Resolution: {metadata.format.columns_per_frame} columns x {metadata.format.pixels_per_column} pixels")
# print(f"UDP Profile: {metadata.format.udp_profile_lidar}")
# print("\nBeam Configuration:")
# print(f"Azimuth Window: {metadata.format.column_window}")
# print(f"Beam Altitude Angles: {metadata.beam_altitude_angles[:5]}...")  # First 5 angles
# print(f"Beam Azimuth Angles: {metadata.beam_azimuth_angles[:5]}...")



source_iter = iter(source)
ctr = 0  # Ensure the counter is initialized before using it
for scan in source_iter:
    if scan:  # Ensure the scan object is valid
        # Retrieve the RANGE field data
        ranges = scan.field(client.ChanField.RANGE)
        print('the values of ranges')
        print(ranges)

        # Increment the counter
        ctr += 1
        

        # Destagger the ranges using metadata for proper spatial alignment
        ranges_destaggered = client.destagger(metadata, ranges)
        print(ranges)
        print(f"Size of range_data: {ranges.shape}")

        # Visualize the destaggered ranges as a grayscale image
        # plt.imshow(ranges_destaggered, cmap='gray', interpolation='nearest')
        # plt.title(f"Destaggered Range - Scan {ctr}")
        # plt.colorbar(label='Range (meters)')
        # plt.savefig(f"scan_{ctr}.png")
        # print(f"Saved scan {ctr} to file.")

        # plt.show()

        # Stop after processing 3 scans (optional)
        if ctr == 3:
            break
    
    

# print("Available fields and corresponding dtype in LidarScan")
# for field in scan.fields:
#     print('{0:11} {1}'.format(str(field), scan.field(field).dtype))
# pose = scan.packet_count
# print(pose)
xyzlut = client.XYZLut(metadata)
range = xyzlut(scan.field(client.ChanField.RANGE))
print(range)

print("values of xyzlut")
print(f"Size of range_data: {range.shape}")













