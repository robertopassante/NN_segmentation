import os
import json
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, Subset
from torchgeo.datasets import LoveDA, LandCoverAI, DeepGlobeLandCover, ChesapeakeCVPR, OpenEarthMap
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
        
        # Flag: se True, il subset è già stato applicato e si salta la logica generica
        _subset_applied = False

        if self.dataset_name == "loveda":
            self.geo_dataset = LoveDA(
                root=self.data_dir,
                split=self.split, 
                download=True, 
                checksum=False
            )
        elif self.dataset_name == "landcoverai":
            self.geo_dataset = LandCoverAI(
                root=self.data_dir, 
                split=self.split, 
                download=True, 
                checksum=False
            )
        elif self.dataset_name == "openearthmap":
            self.geo_dataset = OpenEarthMap(
                root=self.data_dir,
                split=self.split,
                download=False,  # Scaricato manualmente da Drive
                checksum=False
            )

            # ── Smart Subset: carica gli indici pre-calcolati da prepare_dataset.py ──
            indices_file = os.path.join(self.data_dir, f"oem_{split}_indices.json")
            if os.path.exists(indices_file):
                with open(indices_file, "r", encoding="utf-8") as f:
                    index_data = json.load(f)

                indices = [s["idx"] for s in index_data["samples"]]
                self.geo_dataset = Subset(self.geo_dataset, indices)
                _subset_applied = True

                print(f"[OEM SMART SUBSET] Split '{split}': {len(indices)} immagini selezionate")
                print(f"  Soglia dominanza : {index_data.get('threshold', '?') * 100:.0f}%")
                print(f"  Distribuzione per classe:")
                class_names = index_data.get("class_names", {})
                for k, cnt in index_data.get("class_counts", {}).items():
                    name = class_names.get(k, f"Classe {k}")
                    if cnt > 0:
                        print(f"    [{k}] {name:<12}: {cnt:>3}")
            else:
                print(f"[OEM] Nessun file indici trovato ({indices_file}).")
                print(f"      Esegui data/prepare_dataset.py prima del training per il subset intelligente.")
                print(f"      Uso fallback: subset casuale con MAX_TRAIN/VAL_SAMPLES.")

        elif self.dataset_name == "deepglobe":
            dg_split = "valid" if self.split == "val" else self.split
            self.geo_dataset = DeepGlobeLandCover(
                root=os.path.join(self.data_dir, "deepglobe"), 
                split=dg_split
            )
        elif self.dataset_name == "chesapeake":
            cp_split = f"va-{self.split}" 
            self.geo_dataset = ChesapeakeCVPR(
                root=os.path.join(self.data_dir, "chesapeake"), 
                splits=[cp_split], 
                layers=["naip-new", "lc"], 
                download=False
            )
        else:
            raise ValueError(f"Dataset {self.dataset_name} non implementato nel wrapper.")
        
        # ── Subset generico (casuale) — usato se il smart subset non è stato applicato ──
        if not _subset_applied:
            max_samples = None
            if split == "train" and Config.MAX_TRAIN_SAMPLES is not None:
                max_samples = Config.MAX_TRAIN_SAMPLES
            elif split == "val" and Config.MAX_VAL_SAMPLES is not None:
                max_samples = Config.MAX_VAL_SAMPLES
                
            if max_samples is not None and len(self.geo_dataset) > max_samples:
                indices = np.random.RandomState(42).choice(
                    len(self.geo_dataset), size=max_samples, replace=False
                )
                self.geo_dataset = Subset(self.geo_dataset, indices)
                print(f"[SUBSET] Usando {max_samples}/{len(self.geo_dataset) + max_samples} immagini per {split}")
            else:
                print(f"[INFO] Usando tutte le {len(self.geo_dataset)} immagini per {split}")
            
    def __len__(self):
        return len(self.geo_dataset)

    def __getitem__(self, idx):
        # torchgeo returns a dictionary
        sample = self.geo_dataset[idx]
        image = sample["image"] # (C, H, W) tensor
        mask = sample["mask"]   # (H, W) or (1, H, W) tensor
        
        # Chesapeake NAIP ha 4 canali (R, G, B, NIR). Prendiamo solo i primi 3 per il modello
        if self.dataset_name == "chesapeake" and image.shape[0] == 4:
            image = image[:3]
        
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
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask_np).long()

        return image_tensor, mask_tensor
