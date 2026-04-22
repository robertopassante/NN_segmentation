"""
dataset_kaggle.py — Dataset custom per OpenEarthMap su Kaggle

Legge direttamente dalla struttura del dataset Kaggle:
  /kaggle/input/global-land-cover-mapping-openearthmap/
  ├── images/
  │   ├── train/   ← aachen_1.tif, roma_5.tif, ...
  │   └── val/
  ├── label/
  │   ├── train/   ← stesse filename, mask con indici 0-8
  │   └── val/
  ├── train.txt    ← lista di filename (solo nome, senza path)
  └── val.txt

Non usa torchgeo (struttura diversa dalla versione ZIP ufficiale).
Gli indici del smart subset vengono letti da dataset/oem_{split}_indices.json
se disponibili, altrimenti usa tutto il dataset.
"""

import os
import json
import torch
import numpy as np
import rasterio
import pywt
import cv2
from torch.utils.data import Dataset, Subset
from config_kaggle import ConfigKaggle as Config


class OEMKaggleDataset(Dataset):
    """
    Dataset PyTorch per OpenEarthMap nella versione Kaggle
    (dataset aletbm/global-land-cover-mapping-openearthmap).
    """

    def __init__(self, split: str = "train", transform=None):
        """
        Args:
            split: "train" o "val"
            transform: pipeline Albumentations (image+mask)
        """
        self.split     = split
        self.transform = transform

        self.images_dir = os.path.join(Config.IMAGES_DIR, split)
        self.labels_dir = os.path.join(Config.LABELS_DIR, split)

        # ── Legge direttamente tutti i file .tif dalla cartella ─────────
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(
                f"Cartella immagini non trovata: {self.images_dir}\n"
                f"Verifica l'allocazione del dataset su Kaggle."
            )

        all_files = [f for f in os.listdir(self.images_dir) if f.endswith('.tif') or f.endswith('.png') or f.endswith('.jpg')]
        self.file_list = sorted(all_files)

        print(f"[OEM-Kaggle] Split '{split}': {len(self.file_list)} immagini caricate per il training/val")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int):
        fname = self.file_list[idx]

        img_path  = os.path.join(self.images_dir, fname)
        mask_path = os.path.join(self.labels_dir, fname)

        # ── Leggi immagine (GeoTIFF RGB) ──────────────────────────────────
        with rasterio.open(img_path) as src:
            # rasterio legge (C, H, W) — prendiamo i primi 3 canali (RGB)
            image_np = src.read(indexes=[1, 2, 3])   # (3, H, W) uint16 o uint8

        # Porta in (H, W, C) uint8 normalizzato 0-255
        image_np = image_np.transpose(1, 2, 0)
        if image_np.dtype != np.uint8:
            # Normalizza da uint16 a uint8
            image_np = (image_np / image_np.max() * 255).clip(0, 255).astype(np.uint8) \
                       if image_np.max() > 0 else image_np.astype(np.uint8)

        # ── Leggi mask (GeoTIFF singolo canale, valori 0-8) ───────────────
        with rasterio.open(mask_path) as src:
            mask_np = src.read(1).astype(np.uint8)   # (H, W)

        # ── Aggiungi Wavelet Edge Channel se abilitato ────────────────────
        if Config.USE_WAVELET_AUGMENTATION:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            # Estraiamo le features ad alta frequenza con la wavelet scelta
            LL, (LH, HL, HH) = pywt.dwt2(gray, Config.WAVELET_TYPE)
            # Combiniamo i contorni
            edges = np.abs(LH) + np.abs(HL) + np.abs(HH)
            # La DWT dimezza la risoluzione, quindi la riportiamo alla grandezza originale
            edges = cv2.resize(edges, (image_np.shape[1], image_np.shape[0]))
            edges = np.clip(edges, 0, 255).astype(np.uint8)
            # Creiamo l'immagine a 4 canali: RGB + Edges
            image_np = np.dstack([image_np, edges])

        # ── Applica trasformazioni Albumentations ─────────────────────────
        if self.transform is not None:
            augmented = self.transform(image=image_np, mask=mask_np)
            image_np  = augmented["image"]
            mask_np   = augmented["mask"]

        # ── Converti in tensor ────────────────────────────────────────────
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
        mask_tensor  = torch.from_numpy(mask_np).long()

        return image_tensor, mask_tensor
