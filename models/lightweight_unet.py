import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class LightweightUNet(nn.Module):
    def __init__(self, num_classes=2, encoder_name="tu-swin_tiny_patch4_window7_224"):
        """
        Initializes a lightweight U-Net using a Swin Transformer backbone (or standard CNN).
        Requires `segmentation-models-pytorch` and `timm`.
        
        Args:
            num_classes (int): Number of segmentation classes.
            encoder_name (str): The timm backbone to use. Defaults to a Swin Tiny transformer.
                                Another extremely fast alternative: "resnet34".
        """
        super().__init__()
        print(f"Initializing U-Net with backbone: {encoder_name}")
        
        # When using purely CPU or testing, some timm Swin versions can be heavy to download.
        # smp handles the downloading and wrapping automatically.
        # "tu-" prefix tells smp to fetch the model from the `timm` library.
        try:
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights="imagenet", # Pre-trained on ImageNet for faster convergence
                in_channels=3,
                classes=num_classes,
            )
            print("Successfully loaded the model backend from smp.")
        except Exception as e:
            print(f"Could not load requested backbone {encoder_name}: {e}")
            print("Falling back to standard ResNet-34 backbone for safety.")
            self.model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights="imagenet",
                in_channels=3,
                classes=num_classes,
            )
            
    def forward(self, x):
        # The smp model takes [B, C, H, W] and outputs [B, num_classes, H, W] directly,
        # perfectly matching your target labels. No need to manually resize feature maps!
        return self.model(x)
