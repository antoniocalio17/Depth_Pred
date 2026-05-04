"""
SMP Unet++ inference for a single image.
"""

import argparse
import os

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torchvision.transforms.v2 as v2


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model() -> torch.nn.Module:
    # Keep these EXACTLY consistent with training.
    return smp.UnetPlusPlus(
        encoder_name="timm-efficientnet-b0",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )


def load_state_dict(model: torch.nn.Module, ckpt_path: str, map_location: torch.device) -> None:
    state = torch.load(ckpt_path, map_location=map_location)

    # tolerate {"state_dict": ...} wrappers
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. missing={missing} unexpected={unexpected}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Input image path")
    p.add_argument("--weights", required=True, help="best_model.pth (state_dict)")
    p.add_argument("--out", default="", help="Output mask path (.png). Default: <image>_mask.png")
    p.add_argument("--thresh", type=float, default=0.9)
    p.add_argument("--size", type=int, default=224)
    args = p.parse_args()

    device = pick_device()
    print(f"[info] device={device}")

    eval_image_transform = v2.Compose(
        [
            v2.Resize((args.size, args.size)),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    model = build_model().to(device)
    load_state_dict(model, args.weights, map_location=device)
    model.eval()
    print("[info] Model loaded and set to evaluation mode.")

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    x = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    x = eval_image_transform(x).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)

        # Expected shapes: (1,1,H,W) or (1,H,W) depending on SMP settings
        prob = prob.squeeze(0).squeeze(0)  # -> (H,W)
        prob = prob.detach().float().cpu().numpy()

    mask = (prob > args.thresh).astype(np.uint8) * 255
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    out_path = args.out
    if not out_path:
        root, _ = os.path.splitext(args.image)
        out_path = root + "_mask.png"

    cv2.imwrite(out_path, mask)
    print(f"[ok] wrote: {out_path}")


if __name__ == "__main__":
    main()