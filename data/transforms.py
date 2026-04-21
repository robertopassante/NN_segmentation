import albumentations as A
import numpy as np
import pywt
import cv2

def apply_wavelet_fusion(image, wavelet='haar'):
    """
    Advanced Strategy inspired by ISPAMM Lab Wavelet research.
    Applies Discrete Wavelet Transform to extract high-frequency details
    and fuses them to enhance edge features (useful for building/crop boundaries).
    """
    # Convert RGB to Grayscale for wavelet transform
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 2D DWT
    coeffs2 = pywt.dwt2(gray, wavelet)
    LL, (LH, HL, HH) = coeffs2
    
    # We can amplify high frequencies (edges/textures)
    LH = LH * 1.5 
    HL = HL * 1.5
    HH = HH * 1.5
    
    # Inverse DWT
    enhanced_gray = pywt.idwt2((LL, (LH, HL, HH)), wavelet)
    enhanced_gray = np.clip(enhanced_gray, 0, 255).astype(np.uint8)
    
    # Merge the enhanced details back to the luma channel (converting to LAB space)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Resize enhanced gray to match image shape due to DWT padding
    enhanced_gray = cv2.resize(enhanced_gray, (lab.shape[1], lab.shape[0]))
    
    lab[:,:,0] = enhanced_gray
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return result

def get_train_transforms(image_size, use_wavelet=False):
    transforms = []
    
    if use_wavelet:
        transforms.append(
            A.Lambda(
                name="wavelet_fusion",
                image=lambda img, **kwargs: apply_wavelet_fusion(img, 'haar'),
                p=1.0 # Always apply if enabled
            )
        )
        
    transforms.extend([
        A.RandomCrop(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return A.Compose(transforms)

def get_val_transforms(image_size, use_wavelet=False):
    transforms = []
    if use_wavelet:
        transforms.append(
            A.Lambda(
                name="wavelet_fusion",
                image=lambda img, **kwargs: apply_wavelet_fusion(img, 'haar'),
                p=1.0
            )
        )
    transforms.extend([
        A.CenterCrop(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return A.Compose(transforms)
