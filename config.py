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
    DATASET_NAME = "openearthmap" # Opzioni: "loveda", "landcoverai", "deepglobe", "chesapeake", "openearthmap"
    
    # Subset: limita il numero di immagini per velocizzare il training (None = usa tutto)
    MAX_TRAIN_SAMPLES = 3000
    MAX_VAL_SAMPLES = 600

    # --- Smart Subset (OpenEarthMap) ---
    # Soglia minima di pixel dominanti per classificare un'immagine in una classe
    DOMINANT_CLASS_THRESHOLD = 0.40  # 40%
    # Numero massimo di immagini selezionate per classe
    SAMPLES_PER_CLASS_TRAIN = 150   # ~1200 immagini totali in train
    SAMPLES_PER_CLASS_VAL   = 40    # ~320 immagini totali in val
    # Classi target per la visualizzazione a batch (4 righe fisse)
    # [6=Water, 8=Building, 5=Tree, 7=Agriculture]
    OEM_VIZ_CLASSES = [6, 8, 5, 7]
    
    # Adattamento Dinamico delle Classi
    if DATASET_NAME == "loveda":
        NUM_CLASSES = 8 # 7 classi originali + 1 background
    elif DATASET_NAME == "landcoverai":
        NUM_CLASSES = 5 # 4 classi + 1 background
    elif DATASET_NAME == "deepglobe":
        NUM_CLASSES = 7 # 6 classi (Urban, Forest, Water...) + 1 unknown
    elif DATASET_NAME == "chesapeake":
        NUM_CLASSES = 7 # Water, Tree Canopy, Field, Barren, Impervious Surface, Impervious Road, No-Data
    elif DATASET_NAME == "openearthmap":
        NUM_CLASSES = 9 # 8 classi (Bareland, Rangeland, Developed, Road, Tree, Water, Agriculture, Building) + 1 background
    else:
        NUM_CLASSES = 2
    
    # Training Parameters
    BATCH_SIZE = 32 # Increased to fully saturate the 15GB T4 GPU VRAM
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 30 # Balanced per testare la convergenza (+75%) in tempi record su Colab
    IMAGE_SIZE = 224 # Resize size for inputs (Forced to 224 for Swin Transformer strict patch embedding requirement)
    
    # Wavelet parameters (ISPAMM integration)
    USE_WAVELET_AUGMENTATION = True
    WAVELET_TYPE = 'haar'
    
    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
