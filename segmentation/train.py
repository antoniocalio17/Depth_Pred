"""
Training loop for foot segmentation models.
"""

import csv
import os

import torch
import mlflow

import cv2

from .visualization import save_epoch_visualizations
from .config import eval_mask_transform, eval_image_transform



vis_images_paths = [r"/data/adat/SAM2 Labels Foot Segmentation/Foot Images/Brighton Penalties/53/2026-04-10_frame_66_left.png",
                    r"/data/adat/SAM2 Labels Foot Segmentation/Foot Images/Brighton Penalties/53/2026-04-10_frame_93_left.png",
                    r"/data/adat/SAM2 Labels Foot Segmentation/Foot Images/Brighton Penalties/1/2025-07-20_frame_86_left.png",
                    r"/data/adat/SAM2 Labels Foot Segmentation/Foot Images/FCM/11/frame_699_right.png",
                    r"/data/adat/SAM2 Labels Foot Segmentation/Foot Images/FCM/12/frame_24_right.png",
                    r"/data/adat/SAM2 Labels Foot Segmentation/Foot Images/FCM/3/frame_77_right.png",
                    r"/data/adat/SAM2 Labels Foot Segmentation/Foot Images/Predag Dual Radar Test/16/radar_1_frame_286_right.png",  
                    r"/data/adat/SAM2 Labels Foot Segmentation/Foot Images/Predag Dual Radar Test/29/radar_1_frame_363_left.png",
                    r"/data/adat/SAM2 Labels Foot Segmentation/Foot Images/Predag Dual Radar Test/41/radar_1_frame_172_left.png"]


vis_input_tensors, vis_segmentations_tensors = [], []

for path in vis_images_paths:
    segmentation_path = path.replace("Foot Images", "Foot Masks")
    image = cv2.imread(path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    segmentation = cv2.imread(segmentation_path, cv2.IMREAD_GRAYSCALE)

    image_rgb = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    segmentation = torch.from_numpy(segmentation).unsqueeze(0).float() / 255.0

    image_rgb = eval_image_transform(image_rgb)
    segmentation = eval_mask_transform(segmentation)
    segmentation = (segmentation > 0.5).float()
    
    vis_input_tensors.append(image_rgb)
    vis_segmentations_tensors.append(segmentation)

vis_input_tensors = torch.stack(vis_input_tensors)
vis_segmentations_tensors = torch.stack(vis_segmentations_tensors)

def train_model(
    model,
    optimizer,
    device,
    train_loader,
    val_loader,
    criterion,
    threshold=0.5,
    max_epochs=50,
    patience=3,
    save_path=None,
    vis_dir=None,
    csv_log_path=None,
    sample_indices=None,
    mlflow_enabled=True,
):
    """
    Train a segmentation model with early stopping and MLFlow logging.
    
    Args:
        model: PyTorch segmentation model
        optimizer: Optimizer instance
        device: torch.device to train on
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        criterion: Loss function
        threshold: Probability threshold for IoU calculation
        max_epochs: Maximum number of training epochs
        patience: Early stopping patience (epochs without improvement)
        save_path: Path to save best model checkpoint
        vis_dir: Directory to save epoch visualizations (optional)
        csv_log_path: Path to CSV file where per-epoch metrics are saved
        sample_indices: Indices of samples to visualize (optional, uses first batch if None)
        mlflow_enabled: Whether to log metrics/artifacts to MLflow
        
    Returns:
        Best IoU score achieved during training
    """
    best_iou = 0.0
    epochs_without_improvement = 0

    if csv_log_path is not None:
        os.makedirs(os.path.dirname(csv_log_path), exist_ok=True)
        with open(csv_log_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "val_iou",
                "best_iou",
                "threshold",
                "epochs_without_improvement",
                "checkpoint_saved",
            ])

    if vis_dir is not None:
        with torch.no_grad():
            vis_logits = model(vis_input_tensors.to(device))
            vis_logits = vis_logits.cpu()


        save_epoch_visualizations(
            images=vis_input_tensors,
            gt_masks=vis_segmentations_tensors,
            pred_logits=vis_logits,
            epoch=-1,
            vis_dir=vis_dir,
            threshold=threshold,
            alpha=0.5,
            max_samples=8)


    for epoch in range(max_epochs):
        
        # -----------------------
        # TRAINING PHASE
        # -----------------------
        model.train()
        train_loss = 0.0
        counter_batch = 0

        for images, masks, masks_loss in train_loader:
            images, masks, masks_loss = images.to(device), masks.to(device), masks_loss.to(device)

            optimizer.zero_grad()
            logits = model(images)

            loss = criterion(logits, masks, masks_loss)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            counter_batch += 1
            
        train_loss /= counter_batch

        # -----------------------
        # VALIDATION PHASE
        # -----------------------
        model.eval()
        val_loss = 0.0
        val_iou_sum = 0.0
        val_iou_count = 0
        

        with torch.no_grad():
            for batch_idx, (images, masks, masks_loss) in enumerate(val_loader):
                images, masks, masks_loss = images.to(device), masks.to(device), masks_loss.to(device)

                logits = model(images)
                loss = criterion(logits, masks,masks_loss)
                val_loss += loss.item() * images.size(0)

                probs = torch.sigmoid(logits)
                preds = (probs > threshold).float()

                preds_flat = preds.view(preds.size(0), -1)
                masks_flat = masks.view(masks.size(0), -1)
                masks_loss_flat = masks_loss.view(masks_loss.size(0), -1)

                # Evaluate IoU only on valid pixels selected by masks_loss.
                valid = (masks_loss_flat > 0.5).float()
                valid_pixels_per_sample = valid.sum(dim=1)
                has_valid = valid_pixels_per_sample > 0

                if has_valid.any():
                    preds_masked = preds_flat * valid
                    masks_masked = masks_flat * valid

                    intersection = (preds_masked * masks_masked).sum(dim=1)
                    union = (preds_masked + masks_masked - preds_masked * masks_masked).sum(dim=1)

                    iou = (intersection + 1e-6) / (union + 1e-6)
                    val_iou_sum += iou[has_valid].sum().item()
                    val_iou_count += int(has_valid.sum().item())
                
               

        val_loss /= len(val_loader.dataset)
        val_iou = val_iou_sum / max(val_iou_count, 1)

        # -----------------------
        # LOGGING
        # -----------------------
        if mlflow_enabled:
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_iou", val_iou, step=epoch)
            mlflow.log_param("threshold", threshold)

        print(
            f"Epoch {epoch} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val IoU: {val_iou:.4f} | "
            f"Threshold: {threshold}"
        )
        
        # -----------------------
        # SAVE VISUALIZATIONS
        # -----------------------


        if vis_dir is not None:

            with torch.no_grad():
                vis_logits = model(vis_input_tensors.to(device))
                vis_logits = vis_logits.cpu()


            save_epoch_visualizations(
                images=vis_input_tensors,
                gt_masks=vis_segmentations_tensors,
                pred_logits=vis_logits,
                epoch=epoch,
                vis_dir=vis_dir,
                threshold=threshold,
                alpha=0.5,
                max_samples=8)

        # -----------------------
        # EARLY STOPPING (based on IoU)
        # -----------------------
        checkpoint_saved = False

        if val_iou > best_iou:
            best_iou = val_iou
            epochs_without_improvement = 0

            if save_path is not None:
                torch.save(model.state_dict(), save_path)
                checkpoint_saved = True
                if mlflow_enabled:
                    mlflow.log_artifact(save_path)

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")

        if csv_log_path is not None:
            with open(csv_log_path, "a", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([
                    epoch,
                    train_loss,
                    val_loss,
                    val_iou,
                    best_iou,
                    threshold,
                    epochs_without_improvement,
                    int(checkpoint_saved),
                ])

        if epochs_without_improvement >= patience:
            break

    torch.cuda.empty_cache()

    # Optuna maximizes this value
    return best_iou
