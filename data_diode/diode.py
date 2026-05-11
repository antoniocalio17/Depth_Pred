"""
diode_outdoor.py
Dataset loader per DIODE (outdoor split), pensato per fine-tuning del decoder
con range limitato a 20m.

Struttura attesa dopo `tar -xzf train.tar.gz` o `tar -xzf val.tar.gz`:
    DIODE_root/
        train/  (o val/)
            outdoor/ (o outdoors/)
                scene_*/
                    scan_*/
                        *.png             RGB 1024x768
                        *_depth.npy       depth in metri (float32)
                        *_depth_mask.npy  validity mask (uint8, 1=valid)

Filtra automaticamente i pixel oltre `max_depth` cosi' il decoder non
deve imparare ad estrapolare verso l'infinito.
"""

import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class DiodeOutdoor(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",                 # "train" o "val"
        image_size: tuple = (518, 518),       # DINOv2 vuole multipli di 14
        min_depth: float = 0.6,
        max_depth: float = 20.0,
        augment: bool = True,
    ):
        super().__init__()
        self.root = root
        self.image_size = image_size
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.augment = augment and split == "train"

        outdoor_dir = os.path.join(root, split, "outdoor")
        if not os.path.isdir(outdoor_dir):
            # Backward compatibility with previous naming.
            outdoor_dir = os.path.join(root, split, "outdoors")
        if not os.path.isdir(outdoor_dir):
            raise FileNotFoundError(
                f"Non trovo {os.path.join(root, split, 'outdoor')} "
                f"(o {os.path.join(root, split, 'outdoors')}). "
                "Hai scompattato train.tar.gz / val.tar.gz?"
            )

        # Trovo tutti i sample camminando la struttura scene/scan
        self.samples = sorted(glob.glob(
            os.path.join(outdoor_dir, "scene_*", "scan_*", "*.png")
        ))
        if not self.samples:
            raise RuntimeError(f"Nessun PNG trovato in {outdoor_dir}")

        print(f"[DiodeOutdoor] {split}: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        rgb_path = self.samples[idx]
        depth_path = rgb_path.replace(".png", "_depth.npy")
        mask_path = rgb_path.replace(".png", "_depth_mask.npy")

        # --- RGB ---
        rgb = Image.open(rgb_path).convert("RGB")
        rgb = TF.resize(rgb, self.image_size, antialias=True)
        rgb_t = TF.to_tensor(rgb)
        rgb_t = TF.normalize(
            rgb_t,
            mean=[0.485, 0.456, 0.406],   # ImageNet, richiesto da DINOv2
            std=[0.229, 0.224, 0.225],
        )

        # --- Depth ---
        depth = np.load(depth_path).astype(np.float32)
        if depth.ndim == 3:
            depth = depth.squeeze(-1)
        depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        depth_t = torch.nn.functional.interpolate(
            depth_t, size=self.image_size, mode="nearest"
        ).squeeze(0)  # (1, H, W)

        # --- Validity mask DIODE ---
        mask = np.load(mask_path).astype(np.uint8)
        if mask.ndim == 3:
            mask = mask.squeeze(-1)
        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        mask_t = torch.nn.functional.interpolate(
            mask_t, size=self.image_size, mode="nearest"
        ).squeeze(0).bool()  # (1, H, W)

        # --- Filtro range: il pezzo chiave ---
        # Pixel oltre max_depth diventano invalidi -> non contribuiscono alla loss.
        # Cosi' il decoder impara a non saturare oltre 20m senza estrapolazione.
        range_mask = (depth_t > self.min_depth) & (depth_t < self.max_depth)
        valid = mask_t & range_mask & torch.isfinite(depth_t)

        # Augmentation leggera (flip orizzontale)
        if self.augment and torch.rand(1).item() < 0.5:
            rgb_t = TF.hflip(rgb_t)
            depth_t = TF.hflip(depth_t)
            valid = TF.hflip(valid)

        return {
            "rgb": rgb_t,           # (3, H, W) normalizzato ImageNet
            "depth": depth_t,       # (1, H, W) in metri
            "valid": valid,         # (1, H, W) bool
            "path": rgb_path,
        }


if __name__ == "__main__":
    # smoke test: aggiusta il path alla tua estrazione
    DIODE_ROOT = "./DIODE"   # contiene val/outdoor/scene_*/scan_*/...

    ds = DiodeOutdoor(DIODE_ROOT, split="val", image_size=(518, 518))
    sample = ds[0]
    print("rgb:  ", sample["rgb"].shape, sample["rgb"].dtype)
    print("depth:", sample["depth"].shape, sample["depth"].dtype)
    print("valid:", sample["valid"].shape, "valid pixels:",
          sample["valid"].sum().item(),
          f"({100 * sample['valid'].float().mean().item():.1f}%)")
    print("depth range on valid:",
          sample["depth"][sample["valid"]].min().item(),
          "->",
          sample["depth"][sample["valid"]].max().item(), "m")