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
    MODEL_TYPE = "vit_b"
    NUM_CLASSES = 8 # LoveDA uses 7 classes + background (0)
    
    # Training Parameters
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 20
    IMAGE_SIZE = 512 # Resize size for inputs
    
    # Wavelet parameters (ISPAMM integration)
    USE_WAVELET_AUGMENTATION = True
    WAVELET_TYPE = 'haar'
    
    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
