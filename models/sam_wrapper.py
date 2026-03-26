import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry

class SAMFeatureExtractor(nn.Module):
    def __init__(self, model_type="vit_b", checkpoint_path="sam_vit_b_01ec64.pth", device="cuda"):
        super().__init__()
        
        print(f"Loading SAM ({model_type}) from {checkpoint_path}...")
        try:
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            # We only need the image encoder for extracting features
            self.image_encoder = sam.image_encoder
            self.image_encoder.to(device)
            
            # Freeze the SAM encoder to save memory, we only train the classification head!
            for param in self.image_encoder.parameters():
                param.requires_grad = False
                
            print("Successfully loaded SAM image encoder.")
        except Exception as e:
            print(f"Could not load SAM weights: {e}")
            print("Initializing a dummy encoder for dry-run/testing purposes.")
            # Dummy encoder if weights are missing, preventing crash during setup
            self.image_encoder = nn.Conv2d(3, 256, kernel_size=16, stride=16).to(device)
            
        self.device = device
        
    def forward(self, x):
        # SAM expects standard normalization, but we assume inputs are already handled
        # The output of the ViT image encoder in SAM is typically [B, 256, 64, 64] for 1024x1024
        # SAM requires 1024x1024 input due to fixed positional embeddings
        x = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)
        features = self.image_encoder(x)
        return features
