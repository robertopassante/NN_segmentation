import torch
import os

class ConfigKaggle:
    """
    Configurazione per l'ambiente Kaggle.
    Usa i path di /kaggle/input/ per il dataset e /kaggle/working/ per gli output.
    
    Dataset sorgente: aletbm/global-land-cover-mapping-openearthmap
    (allegato come read-only in /kaggle/input/)
    """

    # ── Paths specifici Kaggle ──────────────────────────────────────────────
    # Dataset raw (read-only, allegato da Kaggle). Auto-detect del path
    import glob
    _found = glob.glob('/kaggle/input/**/train.txt', recursive=True)
    KAGGLE_INPUT_DIR = os.path.dirname(_found[0]) if _found else "/kaggle/input/global-land-cover-mapping-openearthmap"
    
    IMAGES_DIR       = os.path.join(KAGGLE_INPUT_DIR, "images")
    LABELS_DIR       = os.path.join(KAGGLE_INPUT_DIR, "label")

    # Cartella di lavoro (read-write) — dove salvare indici, checkpoint, plot
    ROOT_DIR  = "/kaggle/working/NN_segmentation"
    DATA_DIR  = os.path.join(ROOT_DIR, "dataset")         # per i JSON degli indici
    TRAIN_DIR = os.path.join(KAGGLE_INPUT_DIR, "images", "train")
    VAL_DIR   = os.path.join(KAGGLE_INPUT_DIR, "images", "val")

    # Slug del dataset Kaggle opzionale con gli indici pre-calcolati
    # (allegalo da "+ Add Data" → cerca "oem-indices")
    INDICES_KAGGLE_INPUT = "/kaggle/input/oem-indices"

    # ── Model Settings ──────────────────────────────────────────────────────
    SAM_CHECKPOINT_PATH = "/kaggle/working/NN_segmentation/sam_vit_b_01ec64.pth"
    ENCODER_NAME        = "tu-swin_tiny_patch4_window7_224"
    DATASET_NAME        = "openearthmap"

    # ── Smart Subset (stesso di config.py) ──────────────────────────────────
    MAX_TRAIN_SAMPLES = 3000
    MAX_VAL_SAMPLES   = 600

    DOMINANT_CLASS_THRESHOLD = 0.40   # 40%
    SAMPLES_PER_CLASS_TRAIN  = 150    # ~1200 immagini totali train
    SAMPLES_PER_CLASS_VAL    = 40     # ~320  immagini totali val

    # Classi target per la visualizzazione (4 righe fisse nel batch plot)
    # [6=Water, 8=Building, 5=Tree, 7=Agriculture]
    OEM_VIZ_CLASSES = [6, 8, 5, 7]

    # ── Classi OpenEarthMap ─────────────────────────────────────────────────
    NUM_CLASSES = 9   # 0=Background + 8 classi semantiche

    # ── Training Parameters ─────────────────────────────────────────────────
    BATCH_SIZE    = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS    = 30
    IMAGE_SIZE    = 224   # Richiesto da Swin Transformer (patch 4x4 su 224x224)

    # ── Wavelet parameters (ISPAMM integration) ─────────────────────────────
    USE_WAVELET_AUGMENTATION = True
    WAVELET_TYPE = 'haar'

    # ── Hardware ────────────────────────────────────────────────────────────
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
