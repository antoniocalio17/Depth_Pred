# test_encoder.py
# test the encoder with nyu data 
import h5py
import numpy as np
import cv2
import torch
from encoder import DaV2Encoder

MAT_PATH   = "data/nyuv2/nyu_depth_v2_labeled.mat"
CHECKPOINT = "checkpoints/depth_anything_v2_vitl.pth"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# load 4 images directly from the .mat file
with h5py.File(MAT_PATH, "r") as f:
    images = f["images"][:4]   # (4, 3, 640, 480) uint8
    depths = f["depths"][:4]   # (4, 640, 480)    float32

# (4, 3, 640, 480) → (4, 480, 640, 3)
images = images.transpose(0, 3, 2, 1)

print(f"images: {images.shape}  depth: {depths.shape}")

# preprocess each image
tensors = []
for i in range(4):
    img = cv2.resize(images[i], (518, 518), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    tensors.append(torch.from_numpy(img).permute(2, 0, 1))

batch = torch.stack(tensors)   # (4, 3, 518, 518)
print(f"batch: {batch.shape}  range: [{batch.min():.2f}, {batch.max():.2f}]")

# --- encoder ---
encoder = DaV2Encoder(checkpoint_path=CHECKPOINT, encoder_size="vitl", freeze=True)

with torch.no_grad():
    f0, f1, f2, f3 = encoder(batch)

print(f"f0: {tuple(f0.shape)}")
print(f"f1: {tuple(f1.shape)}")
print(f"f2: {tuple(f2.shape)}")
print(f"f3: {tuple(f3.shape)}")