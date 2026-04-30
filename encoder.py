import os
import cv2
import numpy as np
import torch
import torch.nn as nn

from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2


def preprocess_image(image, target_size=518):
    """
    Preprocessing function compatible with DaV2.
    Input:
      - image: either a path string or a BGR np.ndarray (OpenCV format)
    Output:
      - tensor: (1, 3, H, W), float32, ImageNet-normalized
      - original_size: (H, W)
 
    """
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"Immagine non trovata: {image}")
    else:
        img = image.copy()

    original_size = img.shape[:2]  # (H, W)

    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize fisso a 518x518 (scelta semplice per training)
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor, original_size


class DaV2Encoder(nn.Module):
    """
    Encoder DaV2 per custom decoder:
    restituisce 4 feature map multi-scala a 256 canali.

    Output (con input 518x518):
      f0: (B, 256, 148, 148)  # high-res
      f1: (B, 256, 74, 74)
      f2: (B, 256, 37, 37)
      f3: (B, 256, 19, 19)    # low-res, high-semantic
    """

    MODEL_CONFIGS = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    def __init__(self, checkpoint_path, encoder_size="vitl", device=None, freeze=True):
        super().__init__()

        if encoder_size not in self.MODEL_CONFIGS:
            raise ValueError(f"encoder_size non valido: {encoder_size}")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint_path}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Modello completo ufficiale DaV2
        cfg = self.MODEL_CONFIGS[encoder_size]
        model = DepthAnythingV2(**cfg)

        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state, strict=True)

        # DINOv2 backbone, classic ViT encoder
        self.pretrained = model.pretrained      #
        # DPT head used as neck to transform information into feature maps to give to the decoder
        self.depth_head = model.depth_head      
        self.layer_idx = model.intermediate_layer_idx[encoder_size] # are literally indexes for each model type, for example vitl has 4 layers, so layer_idx will be [4, 11, 17, 23] because it has 24 blocks in total

        if freeze:
            for p in self.pretrained.parameters():
                p.requires_grad = False
            for p in self.depth_head.parameters():
                p.requires_grad = False
            self.eval()

        self.to(self.device)
        print(f"[DaV2Encoder] Loaded: {checkpoint_path}")
        print(f"[DaV2Encoder] Encoder: {encoder_size} | Device: {self.device} | Freeze: {freeze}")

    @torch.no_grad()
    def forward(self, x):
        """
        x: (B, 3, H, W), preferibilmente multipli di 14.
        Return: f0, f1, f2, f3 (tutte a 256 canali per vitl)
        """
        x = x.to(self.device) # moves input tensor on the same device of the model
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14 # Vision Transformer breaks up image in 14 x14 patches and treat them as tokens
        # get intermediate information from the ViT, returns token features from selected transformer blocks (self.layer_idx)
        features = self.pretrained.get_intermediate_layers( 
            x, self.layer_idx, return_class_token=True
        )
        # loop through the 4 selected layers and apply the depth head to each one
        outs = []
        for i, feat in enumerate(features):
            if self.depth_head.use_clstoken:
                token, cls_token = feat[0], feat[1]
                readout = cls_token.unsqueeze(1).expand_as(token)
                token = self.depth_head.readout_projects[i](torch.cat((token, readout), -1))
            else:
                token = feat[0]

            token = token.permute(0, 2, 1).reshape(token.shape[0], token.shape[-1], patch_h, patch_w)
            token = self.depth_head.projects[i](token)
            token = self.depth_head.resize_layers[i](token)
            outs.append(token)

        layer_1, layer_2, layer_3, layer_4 = outs

        # Uniforma canali via scratch (come in DPT)
        f0 = self.depth_head.scratch.layer1_rn(layer_1)
        f1 = self.depth_head.scratch.layer2_rn(layer_2)
        f2 = self.depth_head.scratch.layer3_rn(layer_3)
        f3 = self.depth_head.scratch.layer4_rn(layer_4)

        return f0, f1, f2, f3


if __name__ == "__main__":
    CHECKPOINT = "checkpoints/depth_anything_v2_vitl.pth"
    IMAGE_PATH = "/Users/user/Desktop/Depth Model/images/hq_example.png"

    tensor, original_size = preprocess_image(IMAGE_PATH, target_size=518)
    print("Original size:", original_size)
    print("Input tensor :", tuple(tensor.shape))

    encoder = DaV2Encoder(
        checkpoint_path=CHECKPOINT,
        encoder_size="vitl",
        device="cuda" if torch.cuda.is_available() else "cpu",
        freeze=True,
    )

    f0, f1, f2, f3 = encoder(tensor)
    print("f0:", tuple(f0.shape))
    print("f1:", tuple(f1.shape))
    print("f2:", tuple(f2.shape))
    print("f3:", tuple(f3.shape))