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

        # ── Legge la lista di file dal txt ufficiale ──────────────────────
        split_txt = os.path.join(Config.KAGGLE_INPUT_DIR, f"{split}.txt")
        if not os.path.exists(split_txt):
            raise FileNotFoundError(
                f"File split non trovato: {split_txt}\n"
                f"Assicurati di aver allegato il dataset "
                f"'global-land-cover-mapping-openearthmap' a questo notebook."
            )

        with open(split_txt, "r") as f:
            all_files = [line.strip() for line in f if line.strip()]

        self.all_files = all_files
        print(f"[OEM-Kaggle] Split '{split}': {len(all_files)} immagini totali nel dataset")

        # ── Smart Subset: carica indici pre-calcolati se disponibili ──────
        indices_file = os.path.join(Config.DATA_DIR, f"oem_{split}_indices.json")
        if os.path.exists(indices_file):
            with open(indices_file, "r", encoding="utf-8") as f:
                index_data = json.load(f)

            # Gli indici puntano alle posizioni nella lista all_files
            self.file_list = [all_files[s["idx"]] for s in index_data["samples"]]
            print(f"[OEM-Kaggle] ✅ Smart subset caricato: {len(self.file_list)} immagini")
            print(f"   Soglia dominanza : {index_data.get('threshold', 0) * 100:.0f}%")
            class_counts = index_data.get("class_counts", {})
            class_names  = index_data.get("class_names", {})
            for k, cnt in class_counts.items():
                if int(cnt) > 0:
                    name = class_names.get(k, f"Classe {k}")
                    print(f"   [{k}] {name:<12}: {cnt}")
        else:
            # Fallback: usa tutta la lista (o un subset casuale)
            max_samples = (Config.MAX_TRAIN_SAMPLES if split == "train"
                           else Config.MAX_VAL_SAMPLES)
            if max_samples and len(all_files) > max_samples:
                rng = np.random.RandomState(42)
                chosen = rng.choice(len(all_files), size=max_samples, replace=False)
                self.file_list = [all_files[i] for i in sorted(chosen)]
                print(f"[OEM-Kaggle] ⚠️ Nessun JSON indici trovato — "
                      f"subset casuale: {max_samples} immagini")
            else:
                self.file_list = all_files
                print(f"[OEM-Kaggle] ⚠️ Nessun JSON indici trovato — "
                      f"uso tutto il dataset: {len(self.file_list)} immagini")
            print(f"   Esegui prepare_dataset_kaggle.py per il subset intelligente.")

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

        # ── Applica trasformazioni Albumentations ─────────────────────────
        if self.transform is not None:
            augmented = self.transform(image=image_np, mask=mask_np)
            image_np  = augmented["image"]
            mask_np   = augmented["mask"]

        # ── Converti in tensor ────────────────────────────────────────────
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
        mask_tensor  = torch.from_numpy(mask_np).long()

        return image_tensor, mask_tensor
