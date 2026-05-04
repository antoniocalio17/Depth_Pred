import argparse
import os
from dataclasses import dataclass
import cv2
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from scipy.io import loadmat
except Exception:
    loadmat = None
from encoder import DaV2Encoder
from decoder import DepthDecoderHead

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class SiLogLoss(nn.Module):
    def __init__(self, lambd = 0.5, eps = 1e-6):
        super().__init__()
        self.lambd = lambd
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor):
        valid = valid_mask.detach().reshape(-1)
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        if valid.sum() == 0:
            return pred.new_tensor(0.0, requires_grad = True)
        pred_valid = torch.clamp(pred_flat[valid], min = self.eps) # clamp specifies the minimum value of the tensor, we clamp the predicted depth to be at least epsilon avoiding log(0)
        target_valid = torch.clamp(target_flat[valid], min = self.eps)
        diff_log = torch.log(target_valid) - torch.log(pred_valid)
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() - self.lambd * torch.pow(diff_log.mean(), 2))
        return loss.mean()

class NYUv2Dataset(Dataset):
    """
    Reads Dataset NYUv2 from nyu_depth_v2_labeled.mat
    """
    def __init__(self, 
            mat_path: str,
            indices: np.ndarray, 
            target_size: int = 518, 
            d_min: float = 0.01, 
            d_max: float = 12.0) -> None:
        self.mat_path = mat_path
        self.indices = indices
        self.target_size = target_size
        self.d_min = d_min
        self.d_max = d_max
        self._h5 = None

    def _ensure_open(self):
        """
        Read the .mat file if it is not already open
        """
        if self._h5 is None:
            self._h5 = h5py.File(self.mat_path, "r")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self,i):
        """
        Mapping function for the dataset, it returns the RGB image, the depth map and the valid mask
        Args:
            i (int): Index of the item
        Returns:
            tuple: A tuple containing the RGB image, the depth map and the valid mask
                - rgb_t (torch.Tensor): The RGB image
                - depth_t (torch.Tensor): The depth map
                - valid_t (torch.Tensor): The valid mask
        """
        self._ensure_open()
        idx = int(self.indices[i])
        rgb = self._h5["images"][idx].transpose(1, 2, 0)
        depth = self._h5["depths"][idx]
        if depth.shape[0] > depth.shape[1]:
            depth = depth.T

        rgb = cv2.resize(rgb, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)

        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - MEAN) / STD

        valid = (depth > self.d_min) & (depth < self.d_max) & np.isfinite(depth)
        depth = np.clip(depth, self.d_min, self.d_max)
        
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        depth_t = torch.from_numpy(depth).unsqueeze(0).contiguous()
        valid_t = torch.from_numpy(valid).unsqueeze(0).bool().contiguous()
        return rgb_t, depth_t, valid_t

@dataclass
class EpochStats:
    loss: float
    abs_rel: float
    rmse: float

def compute_depth_metrics(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor):
    # pred/target/valid: (B,1,H,W)
    p = pred[valid]
    t = target[valid]
    if p.numel() == 0:
        z = pred.new_tensor(0.0)
        return z, z

    abs_rel = torch.mean(torch.abs(p - t) / torch.clamp(t, min=1e-6))
    rmse = torch.sqrt(torch.mean(torch.pow(p - t, 2)))
    return abs_rel, rmse
    

def load_nyu_splits(splits_path: str, n_total: int):
    if splits_path and os.path.exists(splits_path) and loadmat is not None:
        m = loadmat(splits_path)
        train_key = "trainNdxs" if "trainNdxs" in m else "train_ndxs"
        test_key = "testNdxs" if "testNdxs" in m else "test_ndxs"
        if train_key in m and test_key in m:
            train_idx = m[train_key].reshape(-1).astype(np.int64) - 1 # MATLAB -> Python indexing
            test_idx = m[test_key].reshape(-1).astype(np.int64) - 1 # MATLAB -> Python indexing
            return train_idx, test_idx
    all_idx = np.arange(n_total, dtype=np.int64)
    split = int(0.9 * n_total)
    return all_idx[:split], all_idx[split:]

def run_epoch(
    encoder: DaV2Encoder,
    decoder: DepthDecoderHead,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_size: int,
    is_train: bool
)-> EpochStats:
    decoder.train(is_train)
    total_loss = 0.0
    total_abs_rel = 0.0
    total_rmse = 0.0
    count = 0
    for rgb, depth, valid in loader:
        rgb = rgb.to(device, non_blocking=True)
        depth = depth.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)
        with torch.no_grad():
            f0, f1, f2, f3 = encoder(rgb)
        with torch.set_grad_enabled(is_train):
            pred = decoder(f0, f1, f2, f3, (target_size, target_size))
            loss = criterion(pred, depth, valid)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        abs_rel, rmse = compute_depth_metrics(pred.detach(), depth, valid)
        bsz = rgb.shape[0]
        total_loss += loss.item() * bsz
        total_abs_rel += abs_rel.item() * bsz
        total_rmse += rmse.item() * bsz
        count += bsz
    if count == 0:
        return EpochStats(0.0, 0.0, 0.0)
    return EpochStats(
        loss=total_loss / count,
        abs_rel=total_abs_rel / count,
        rmse=total_rmse / count
    )



def main():
    p = argparse.ArgumentParser("Train decoder+head on NYUv2")
    p.add_argument("--mat_path", type=str, default="data/nyuv2/nyu_depth_v2_labeled.mat")
    p.add_argument("--splits_path", type=str, default="data/nyuv2/splits.mat")
    p.add_argument("--checkpoint", type=str, default="checkpoints/depth_anything_v2_vitl.pth")
    p.add_argument("--encoder_size", type=str, default="vitl", choices=["vits", "vitb", "vitl", "vitg"])
    p.add_argument("--output_dir", type=str, default="runs/nyuv2_decoder_head")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--target_size", type=int, default=518)
    p.add_argument("--d_min", type=float, default=0.1)
    p.add_argument("--d_max", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[info] device={device}")

    with h5py.File(args.mat_path, "r") as f:
        n_total = len(f["images"])
    train_idx, test_idx = load_nyu_splits(args.splits_path, n_total)
    train_dataset = NYUv2Dataset(args.mat_path,train_idx,args.target_size,args.d_min,args.d_max)
    test_dataset = NYUv2Dataset(args.mat_path,test_idx,args.target_size,args.d_min,args.d_max)
    use_pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=use_pin_memory)
    encoder = DaV2Encoder(args.checkpoint,args.encoder_size,device = device,freeze = True)
    decoder = DepthDecoderHead(enc_ch = 256,dec_out_ch = 128,max_depth = args.d_max).to(device)
    criterion = SiLogLoss()
    optimizer = torch.optim.AdamW(decoder.parameters(),lr = args.lr,weight_decay = args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = 10,gamma = 0.1)
    best_abs_rel = float('inf')

    for epoch in range(args.epochs):
        train_stats = run_epoch(
            encoder,
            decoder,
            train_loader,
            criterion,
            optimizer,
            device,
            args.target_size,
            is_train=True
        )
        val_stats = run_epoch(
            encoder,
            decoder,
            test_loader,
            criterion,
            optimizer,
            device,
            args.target_size,
            is_train=False
        )
        scheduler.step()
        print(
            f"[epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_stats.loss:.4f} train_abs_rel={train_stats.abs_rel:.4f} train_rmse={train_stats.rmse:.4f} | "
            f"val_loss={val_stats.loss:.4f} val_abs_rel={val_stats.abs_rel:.4f} val_rmse={val_stats.rmse:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        last_checkpoint = {
            "epoch": epoch,
            "decoder_head": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_abs_rel": val_stats.abs_rel,
            "args": vars(args),
        }
        torch.save(last_checkpoint, os.path.join(args.output_dir, "last.pt"))

        if val_stats.abs_rel < best_abs_rel:
            best_abs_rel = val_stats.abs_rel
            torch.save(last_checkpoint, os.path.join(args.output_dir, "best.pt"))
            print(f"[info] saved best checkpoint (val_abs_rel={best_abs_rel:.4f})")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return best_abs_rel

if __name__ == "__main__":
    main()