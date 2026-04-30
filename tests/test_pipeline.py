import h5py
import numpy as np
import cv2
import torch

from encoder import DaV2Encoder
from decoder import DepthDecoderHead

MAT_PATH = "data/nyuv2/nyu_depth_v2_labeled.mat"
CHECKPOINT = "checkpoints/depth_anything_v2_vitl.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def load_nyu_sample(idx=0):
    with h5py.File(MAT_PATH, "r") as f:
        # images in your repo tests are (N, 3, 640, 480) -> transpose to (N, 480, 640, 3)
        images = f["images"][idx:idx+1].transpose(0, 3, 2, 1)  # (1, H, W, 3)
        depths = f["depths"][idx:idx+1]  # (1, 640, 480) typically
    rgb = images[0]                      # (H, W, 3), uint8
    depth = depths[0]                    # GT depth map
    return rgb, depth

def preprocess_rgb(rgb, target_size=518):
    original_size = rgb.shape[:2]  # (H, W)
    img = cv2.resize(rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1,3,518,518)
    return x, original_size

def main():
    rgb, depth_gt = load_nyu_sample(idx=0)
    x, original_size = preprocess_rgb(rgb, target_size=518)

    encoder = DaV2Encoder(
        checkpoint_path=CHECKPOINT,
        encoder_size="vitl",
        device=DEVICE,
        freeze=True,
    )
    decoder_head = DepthDecoderHead(enc_ch=256, dec_out_ch=128, max_depth=10.0).to(DEVICE)
    decoder_head.eval()

    x = x.to(DEVICE)

    with torch.no_grad():
        f0, f1, f2, f3 = encoder(x)
        pred_depth = decoder_head(f0, f1, f2, f3, original_size=original_size)

    print("Input:", tuple(x.shape))
    print("f0:", tuple(f0.shape))
    print("f1:", tuple(f1.shape))
    print("f2:", tuple(f2.shape))
    print("f3:", tuple(f3.shape))
    print("Pred depth:", tuple(pred_depth.shape))
    print("Pred range [m]:", float(pred_depth.min()), float(pred_depth.max()))
    print("GT depth shape:", depth_gt.shape, "GT range:", float(depth_gt.min()), float(depth_gt.max()))

if __name__ == "__main__":
    main()