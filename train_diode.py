import argparse
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_diode.diode import DiodeOutdoor
from decoder import DepthDecoderHead
from encoder import DaV2Encoder


class SiLogLoss(nn.Module):
    def __init__(self, lambd: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.lambd = lambd
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor):
        valid = valid_mask.detach().reshape(-1)
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        if valid.sum() == 0:
            return pred.new_tensor(0.0, requires_grad=True)

        pred_valid = torch.clamp(pred_flat[valid], min=self.eps)
        target_valid = torch.clamp(target_flat[valid], min=self.eps)
        diff_log = torch.log(target_valid) - torch.log(pred_valid)
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() - self.lambd * torch.pow(diff_log.mean(), 2))
        return loss.mean()


@dataclass
class EpochStats:
    loss: float
    abs_rel: float
    rmse: float


def compute_depth_metrics(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor):
    p = pred[valid]
    t = target[valid]
    if p.numel() == 0:
        z = pred.new_tensor(0.0)
        return z, z
    abs_rel = torch.mean(torch.abs(p - t) / torch.clamp(t, min=1e-6))
    rmse = torch.sqrt(torch.mean(torch.pow(p - t, 2)))
    return abs_rel, rmse


def _batch_to_tensors(batch):
    # data_diode returns a dict with rgb/depth/valid
    rgb = batch["rgb"]
    depth = batch["depth"]
    valid = batch["valid"]
    return rgb, depth, valid


def run_epoch(
    encoder: DaV2Encoder,
    decoder: DepthDecoderHead,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_size: int,
    is_train: bool,
    max_batches: int = -1,
) -> EpochStats:
    decoder.train(is_train)
    total_loss = 0.0
    total_abs_rel = 0.0
    total_rmse = 0.0
    count = 0

    for step, batch in enumerate(loader):
        if max_batches > 0 and step >= max_batches:
            break

        rgb, depth, valid = _batch_to_tensors(batch)
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
        rmse=total_rmse / count,
    )


def main():
    p = argparse.ArgumentParser("Train decoder+head on DIODE outdoor")
    p.add_argument("--diode_root", type=str, default="data_diode/DIODE")
    p.add_argument("--train_split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--val_split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--checkpoint", type=str, default="checkpoints/depth_anything_v2_vitl.pth")
    p.add_argument("--encoder_size", type=str, default="vitl", choices=["vits", "vitb", "vitl", "vitg"])
    p.add_argument("--output_dir", type=str, default="runs/diode_decoder_head")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--target_size", type=int, default=518)
    p.add_argument("--min_depth", type=float, default=0.7)
    p.add_argument("--max_depth", type=float, default=20.0)
    p.add_argument("--max_train_batches", type=int, default=30)
    p.add_argument("--max_val_batches", type=int, default=10)
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

    train_dataset = DiodeOutdoor(
        root=args.diode_root,
        split=args.train_split,
        image_size=(args.target_size, args.target_size),
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        augment=True,
    )
    val_dataset = DiodeOutdoor(
        root=args.diode_root,
        split=args.val_split,
        image_size=(args.target_size, args.target_size),
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        augment=False,
    )

    use_pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=use_pin_memory,
    )

    encoder = DaV2Encoder(args.checkpoint, args.encoder_size, device=device, freeze=True)
    decoder = DepthDecoderHead(
        enc_ch=256,
        dec_out_ch=128,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    ).to(device)
    criterion = SiLogLoss()
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_abs_rel = float("inf")
    for epoch in range(args.epochs):
        train_stats = run_epoch(
            encoder=encoder,
            decoder=decoder,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            target_size=args.target_size,
            is_train=True,
            max_batches=args.max_train_batches,
        )
        val_stats = run_epoch(
            encoder=encoder,
            decoder=decoder,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            target_size=args.target_size,
            is_train=False,
            max_batches=args.max_val_batches,
        )
        scheduler.step()

        print(
            f"[epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_stats.loss:.4f} train_abs_rel={train_stats.abs_rel:.4f} train_rmse={train_stats.rmse:.4f} | "
            f"val_loss={val_stats.loss:.4f} val_abs_rel={val_stats.abs_rel:.4f} val_rmse={val_stats.rmse:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        checkpoint = {
            "epoch": epoch,
            "decoder_head": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_abs_rel": val_stats.abs_rel,
            "args": vars(args),
        }
        torch.save(checkpoint, os.path.join(args.output_dir, "last.pt"))
        if val_stats.abs_rel < best_abs_rel:
            best_abs_rel = val_stats.abs_rel
            torch.save(checkpoint, os.path.join(args.output_dir, "best.pt"))
            print(f"[info] saved best checkpoint (val_abs_rel={best_abs_rel:.4f})")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
