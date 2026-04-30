from mask import Mask
import numpy as np
import os

IMAGE_PATH = "/Users/user/Desktop/Thesis/images/penalty.png"
DEPTH_PATH = "/Users/user/Desktop/Thesis/DA-V2/Depth-Anything-V2/output/penalty_raw_depth_meter.npy"

predictor = Mask()
mask = predictor.mask(image_path=IMAGE_PATH)
mask_coordinates = Mask.get_mask_coordinates(mask)

# Depth .npy must be from metric_depth/run.py --save-numpy for this image
base = os.path.splitext(IMAGE_PATH)[0]
depth_path = DEPTH_PATH
depth_map = np.load(depth_path)

depths_at_mask = np.array([depth_map[i, j] for i, j in mask_coordinates])
final_depth = float(depths_at_mask.mean())   
print(f"Final depth: {final_depth} m")
depths = depths_at_mask
print("min/median/mean/max:", depths.min(), np.median(depths), depths.mean(), depths.max())
print("p10/p50/p90:", np.percentile(depths, [10,50,90]))
