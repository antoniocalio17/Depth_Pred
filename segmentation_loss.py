import torch
import torch.nn as nn
import torch.nn.functional as F


class PersonMaskedBCEDiceLoss(nn.Module):
    """
    Computes segmentation loss only on person/ROI pixels.

    Expected inputs:
      - pred_logits: (B,1,H,W) model logits
      - target_mask: (B,1,H,W) binary GT mask (0/1)
      - person_mask: (B,1,H,W) binary ROI mask (1 = include in loss, 0 = ignore)

    If person_mask is empty for a batch, returns 0.0 (no gradient update for that batch).
    """

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.eps = eps

    def _masked_mean(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x, mask: (B,1,H,W)
        denom = mask.sum()
        if denom <= 0:
            return x.new_tensor(0.0, requires_grad=True)
        return (x * mask).sum() / denom

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_mask: torch.Tensor,
        person_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Ensure float tensors
        target_mask = target_mask.float()
        person_mask = (person_mask > 0.5).float()

        # 1) Masked BCE
        bce_map = F.binary_cross_entropy_with_logits(pred_logits, target_mask, reduction="none")
        bce_loss = self._masked_mean(bce_map, person_mask)

        # 2) Masked Dice
        probs = torch.sigmoid(pred_logits)

        probs_m = probs * person_mask
        target_m = target_mask * person_mask

        intersection = (probs_m * target_m).sum(dim=(1, 2, 3))
        denom = probs_m.sum(dim=(1, 2, 3)) + target_m.sum(dim=(1, 2, 3))
        dice_per_sample = 1.0 - (2.0 * intersection + self.eps) / (denom + self.eps)

        # Only average Dice over samples that have at least one valid person pixel
        valid_per_sample = (person_mask.sum(dim=(1, 2, 3)) > 0).float()
        valid_count = valid_per_sample.sum()
        if valid_count > 0:
            dice_loss = (dice_per_sample * valid_per_sample).sum() / valid_count
        else:
            dice_loss = pred_logits.new_tensor(0.0, requires_grad=True)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss