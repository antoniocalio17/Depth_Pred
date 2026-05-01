# decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def _conv_bn_relu(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class FusionBlock(nn.Module):
    """
    It upsamples the lower level feature map to the same resolution as the higher level.
    It concatenates the lower level feature map with the skip connection from the higher level.
    It applies two convolutions to refine the feature map.

    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv_low = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.refine = nn.Sequential(
            _conv_bn_relu(out_ch + in_ch, out_ch),
            _conv_bn_relu(out_ch, out_ch),
        )

    def forward(self, low, high):
        # low  : (B, in_ch, H/2, W/2)
        # high : (B, in_ch, H,   W)
        # we upsample the low level feature map to the same resolution as the higher level, in fact [-2:] is the last two dimensions of the tensor (H, W)
        low_up = F.interpolate(low, size=high.shape[-2:], mode="bilinear", align_corners=False)
        low_up = self.conv_low(low_up)              # (B, out_ch, H, W)
        x = torch.cat([low_up, high], dim=1)        # (B, out_ch + in_ch, H, W)
        return self.refine(x)                       # (B, out_ch, H, W)


class DepthDecoder(nn.Module):
    """
    FPN bottom-up:
        f3 → bottleneck
           ↓ fuse with f2 → fuse with f1 → fuse with f0
                                               ↓
                                    (B, out_ch, 148, 148)
    
    It takes the feature maps from the encoder and fuses them together to generate a feature map at the desired resolution.
    """
    def __init__(self, enc_ch: int = 256, out_ch: int = 128):
        super().__init__()
        self.bottleneck = _conv_bn_relu(enc_ch, enc_ch)
        self.fuse2 = FusionBlock(enc_ch, enc_ch)        # 19→37
        self.fuse1 = FusionBlock(enc_ch, enc_ch)        # 37→74
        self.fuse0 = FusionBlock(enc_ch, out_ch)        # 74→148

    def forward(self, f0, f1, f2, f3):
        """
        REMIND :
        Since the class inherits from nn.Module, 
        its instances are callable (module(...)), 
        and this triggers __call__, 
        which then runs forward(...) with the provided tensors.
        """
        x = self.bottleneck(f3)     # (B, 256, 19,  19)
        x = self.fuse2(x, f2)      # (B, 256, 37,  37) 
        x = self.fuse1(x, f1)      # (B, 256, 74,  74)
        x = self.fuse0(x, f0)      # (B, 128, 148, 148)
        return x


class DepthHead(nn.Module):
    """
    Converte la feature map del decoder in una depth map metrica in metri.
    Softplus garantisce output sempre positivo.
    """
    def __init__(self, in_ch: int = 128, max_depth: float = 10.0):
        super().__init__()
        self.max_depth = max_depth
        self.head = nn.Sequential(
            _conv_bn_relu(in_ch, 64),
            _conv_bn_relu(64, 32),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        self.act = nn.Softplus()

    def forward(self, features, original_size):
        # features     : (B, in_ch, 148, 148)
        # original_size: (H_orig, W_orig)
        x = self.head(features)             # (B, 1, 148, 148)
        x = self.act(x).clamp(max=self.max_depth)
        x = F.interpolate(x, size=original_size, mode="bilinear", align_corners=False)
        return x                            # (B, 1, H_orig, W_orig) in metri


class DepthDecoderHead(nn.Module):
    """Wrapper unico per training e inference."""
    def __init__(self, enc_ch: int = 256, dec_out_ch: int = 128, max_depth: float = 10.0):
        super().__init__()
        self.decoder = DepthDecoder(enc_ch=enc_ch, out_ch=dec_out_ch)
        self.head    = DepthHead(in_ch=dec_out_ch, max_depth=max_depth)

    def forward(self, f0, f1, f2, f3, original_size):
        features = self.decoder(f0, f1, f2, f3)
        return self.head(features, original_size)