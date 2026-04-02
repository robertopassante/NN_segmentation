import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchgeo.datasets import LoveDA, LandCoverAI
from config import Config

class SatelliteSegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None, split="train"):
        """
        Uses torchgeo's datasets dynamically based on Config.DATASET_NAME
        data_dir: where to store/download the data
        split: "train", "val", or "test"
        """
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        self.dataset_name = Config.DATASET_NAME.lower()
        
        # Ensure base directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        print(f"Initializing TorchGeo {self.dataset_name.upper()} ({split} split)...")
        
        if self.dataset_name == "loveda":
            self.geo_dataset = LoveDA(
                root=self.data_dir, # Per LoveDA manteniamo la root standard per non doverlo riscaricare
                split=self.split, 
                download=True, 
                checksum=False
            )
        elif self.dataset_name == "landcoverai":
            self.geo_dataset = LandCoverAI(
                root=os.path.join(self.data_dir, "landcoverai"), 
                split=self.split, 
                download=True, 
                checksum=False
            )
        else:
            raise ValueError(f"Dataset {self.dataset_name} non implementato nel wrapper.")
            
    def __len__(self):
        return len(self.geo_dataset)

    def __getitem__(self, idx):
        # torchgeo returns a dictionary
        sample = self.geo_dataset[idx]
        image = sample["image"] # (C, H, W) tensor
        mask = sample["mask"]   # (H, W) or (1, H, W) tensor
        
        # Convert to numpy for Albumentations (H, W, C)
        image_np = image.numpy().transpose(1, 2, 0)
        # Normalize to 0-255 if it's not already
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
            
        mask_np = mask.numpy().squeeze().astype(np.uint8)

        # Apply transforms (albumentations)
        if self.transform is not None:
            augmented = self.transform(image=image_np, mask=mask_np)
            image_np = augmented['image']
            mask_np = augmented['mask']
            
        # Convert back to tensor
        # image should be (3, H, W) float
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float() / 255.0
        mask_tensor = torch.from_numpy(mask_np).long()

        return image_tensor, mask_tensor
