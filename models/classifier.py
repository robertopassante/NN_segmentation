import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=2):
        """
        A lightweight convolutional head that takes SAM features and upsamples 
        them back to the original image size for pixel-wise classification.
        """
        super().__init__()
        
        # SAM encoder outputs 256 channels spatial maps
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Output layer
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x, target_size=(512, 512)):
        # x is the deep feature map from SAM: [B, 256, H/16, W/16]
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Upsample intermediate features
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Upsample to target size and predict classes
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        x = self.out_conv(x)
        
        return x

class SAM_Segmenter(nn.Module):
    def __init__(self, sam_extractor, num_classes=2):
        super().__init__()
        self.extractor = sam_extractor
        self.head = SegmentationHead(in_channels=256, num_classes=num_classes)
        
    def forward(self, x):
        # Record original size
        h, w = x.shape[2], x.shape[3]
        
        # Extract deep features without computing gradients for the backbone
        with torch.no_grad():
            features = self.extractor(x)
            
        # Pass features through our trainable classification head
        logits = self.head(features, target_size=(h, w))
        return logits
