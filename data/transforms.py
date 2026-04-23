import albumentations as A
import numpy as np
import pywt
import cv2



def get_train_transforms(image_size, use_wavelet=False):
    transforms = []
    
    # Selettore mean/std per 3 canali (RGB) o 4 canali (RGB + Wavelet)
    mean_vals = (0.485, 0.456, 0.406, 0.5) if use_wavelet else (0.485, 0.456, 0.406)
    std_vals  = (0.229, 0.224, 0.225, 0.5) if use_wavelet else (0.229, 0.224, 0.225)
    
    transforms.extend([
        A.RandomCrop(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # --- STRONG AUGMENTATION (Asymmetric Augmentation per Pseudo-Labels) ---
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=2, min_height=16, min_width=16, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        A.Normalize(mean=mean_vals, std=std_vals),
    ])
    return A.Compose(transforms)

def get_val_transforms(image_size, use_wavelet=False):
    transforms = []
    
    mean_vals = (0.485, 0.456, 0.406, 0.5) if use_wavelet else (0.485, 0.456, 0.406)
    std_vals  = (0.229, 0.224, 0.225, 0.5) if use_wavelet else (0.229, 0.224, 0.225)
    
    transforms.extend([
        A.CenterCrop(image_size, image_size),
        A.Normalize(mean=mean_vals, std=std_vals),
    ])
    return A.Compose(transforms)
