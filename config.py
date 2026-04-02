import torch
import os

class Config:
    # Paths
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT_DIR, "dataset")
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR = os.path.join(DATA_DIR, "val")
    
    # Model Settings
    SAM_CHECKPOINT_PATH = "sam_vit_b_01ec64.pth" # Path to downloaded SAM weights
    ENCODER_NAME = "tu-swin_tiny_patch4_window7_224" # Lightweight Swin Transformer backbone
    NUM_CLASSES = 8 # LoveDA uses 7 classes + background (0)
    
    # Training Parameters
    BATCH_SIZE = 32 # Increased to fully saturate the 15GB T4 GPU VRAM
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 40 # Increased to allow Swin to reach >75% performance
    IMAGE_SIZE = 224 # Resize size for inputs (Forced to 224 for Swin Transformer strict patch embedding requirement)
    
    # Wavelet parameters (ISPAMM integration)
    USE_WAVELET_AUGMENTATION = False
    WAVELET_TYPE = 'haar'
    
    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
