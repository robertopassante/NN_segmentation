"""
main_kaggle.py — Entry point per il training su Kaggle

Versione Kaggle di main.py. Usa:
  - config_kaggle.py  (path /kaggle/)
  - data/dataset_kaggle.py  (lettura diretta GeoTIFF senza torchgeo)
  - models/lightweight_unet.py  (invariato)
  - utils/engine.py  (invariato)
  - utils/plots.py  (invariato)
  
I checkpoint e i plot vengono salvati in /kaggle/working/NN_segmentation/
e sono accessibili dal tab "Output" di Kaggle.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import argparse

from config_kaggle import ConfigKaggle as Config
from data.dataset_kaggle import OEMKaggleDataset
from data.transforms import get_train_transforms, get_val_transforms
from models.lightweight_unet import LightweightUNet
from utils.engine import train_one_epoch, evaluate
from utils.plots import plot_loss_curves, save_predictions
from tqdm import tqdm
import random

def prepare_visualization_pools(dataset, target_classes):
    """
    Scansiona il validation dataset UNA SOLA VOLTA all'inizio e trova
    gli indici delle immagini che contengono i target_classes.
    Restituisce un dizionario {class_idx: [lista di indici nel dataset]}.
    """
    print("\n[VIZ] Scansione validation set per creare i batch di visualizzazione...")
    pools = {c: [] for c in target_classes}
    
    # Campioniamo un sottoinsieme se il validation è grande per fare prima
    indices_to_check = list(range(len(dataset)))
    
    for idx in tqdm(indices_to_check, desc="Scanning val masks"):
        _, mask = dataset[idx]
        mask_np = mask.numpy() if hasattr(mask, 'numpy') else mask
        
        # Aggiunge l'indice se la classe è presente almeno all'5%
        for c in target_classes:
            if (mask_np == c).mean() > 0.05:
                pools[c].append(idx)
                
    for c in target_classes:
        if len(pools[c]) == 0:
            print(f"[WARN] Classe {c} non trovata nel validation set (>5% px). Fallback a random.")
            pools[c] = list(range(len(dataset)))
    return pools



def main(args):
    print("=" * 65)
    print("  NN Segmentation — Training su Kaggle")
    print("=" * 65)
    print(f"Device       : {Config.DEVICE}")
    print(f"Batch size   : {Config.BATCH_SIZE}")
    print(f"Epochs       : {Config.NUM_EPOCHS}")
    print(f"Image size   : {Config.IMAGE_SIZE}x{Config.IMAGE_SIZE}")
    print(f"Classi       : {Config.NUM_CLASSES}")
    print(f"Wavelet Aug  : {'ON' if Config.USE_WAVELET_AUGMENTATION else 'OFF'}")
    print("=" * 65)

    # ── 1. Dataset & DataLoaders ─────────────────────────────────────────
    train_transform = get_train_transforms(
        Config.IMAGE_SIZE, use_wavelet=Config.USE_WAVELET_AUGMENTATION
    )
    val_transform = get_val_transforms(Config.IMAGE_SIZE, use_wavelet=False)

    print("\n[DATA] Caricamento dataset...")
    train_dataset = OEMKaggleDataset(split="train", transform=train_transform)
    val_dataset   = OEMKaggleDataset(split="val",   transform=val_transform)

    print(f"\n[DATA] Train: {len(train_dataset)} immagini | Val: {len(val_dataset)} immagini")

    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True
    )

    # ── 1.5. Prepara i pool per la visualizzazione dinamica ──────────────
    viz_pools = prepare_visualization_pools(val_dataset, Config.OEM_VIZ_CLASSES)

    # ── 2. Modello ───────────────────────────────────────────────────────
    print("\n[MODEL] Inizializzazione modello...")
    model = LightweightUNet(
        num_classes=Config.NUM_CLASSES,
        encoder_name=Config.ENCODER_NAME
    )
    
    # Se abbiamo più di 1 GPU (come le 2 T4 di Kaggle), parallelizziamo!
    if torch.cuda.device_count() > 1:
        print(f"\n🚀 Trovate {torch.cuda.device_count()} GPU! Attivazione DataParallel per dividere il carico...")
        model = torch.nn.DataParallel(model)
        
    model = model.to(Config.DEVICE)

    # ── 3. Loss, Optimizer, Scheduler ────────────────────────────────────
    print("\n[TRAIN] Congelamento backbone Swin-T (warm-up 3 epoche)...")
    # Troviamo i layer del backbone. Se è avvolto in DataParallel c'è .module
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    for param in base_model.model.encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LEARNING_RATE
    )

    import segmentation_models_pytorch as smp_module

    class CombinedLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.ce   = nn.CrossEntropyLoss()
            self.dice = smp_module.losses.DiceLoss(mode='multiclass')
        def forward(self, preds, targets):
            return self.ce(preds, targets) + self.dice(preds, targets)

    criterion = CombinedLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS
    )

    train_losses, val_losses = [], []
    best_miou = 0.0

    # ── 4. Training loop ─────────────────────────────────────────────────
    print("\n[TRAIN] Avvio training loop...")
    for epoch in range(Config.NUM_EPOCHS):

        # Sblocco backbone dopo 3 epoche di warm-up
        if epoch == 3:
            print("\n🔥 SCONGELAMENTO BACKBONE: Inizio Fine-Tuning Profondo!")
            base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            for param in base_model.model.encoder.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(
                model.parameters(), lr=Config.LEARNING_RATE / 10
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=Config.NUM_EPOCHS - 3
            )

        print(f"\n--- Epoch {epoch+1}/{Config.NUM_EPOCHS} ---")

        # Dry-run: esegui solo 1 batch per testare il codice
        if args.dry_run:
            print("[Dry Run] Skipped full epoch calculation.")
            break

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )
        scheduler.step()

        val_loss, val_acc, val_miou, val_dice = evaluate(
            model, val_loader, criterion, Config.DEVICE,
            num_classes=Config.NUM_CLASSES
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc*100:.2f}% | mIoU: {val_miou:.4f} | Dice: {val_dice:.4f}"
        )

        # ── Plot loss curves ──────────────────────────────────────────────
        plot_loss_curves(
            train_losses, val_losses,
            save_path=os.path.join(Config.ROOT_DIR, "loss_curve.png")
        )

        # ── Visualizzazione predizioni per classe (Dinamica) ─────────────
        model.eval()
        with torch.no_grad():
            try:
                # Seleziona 4 indici casuali (uno per ogni target class) dai pool pre-calcolati
                viz_indices = [random.choice(viz_pools[c]) for c in Config.OEM_VIZ_CLASSES]
                
                # Crea tensor batch
                imgs_list, msks_list = [], []
                for idx in viz_indices:
                    img, msk = val_dataset[idx]
                    imgs_list.append(img)
                    msks_list.append(msk)
                    
                val_imgs = torch.stack(imgs_list).to(Config.DEVICE)
                val_msks = torch.stack(msks_list).to(Config.DEVICE)
                
                logits   = model(val_imgs)
                save_predictions(
                    val_imgs, val_msks, logits,
                    save_dir=os.path.join(Config.ROOT_DIR, "output_samples"),
                    epoch=epoch, batch_idx=0,
                    mIoU=val_miou, mDice=val_dice
                )
            except Exception as e:
                print(f"[WARN] Skip visualizzazione: {e}")

        # ── Salva checkpoint (best model) ────────────────────────────────
        ckpt_path = os.path.join(Config.ROOT_DIR, "best_model.pth")
        if val_miou > best_miou:
            best_miou = val_miou
            base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            torch.save(base_model.state_dict(), ckpt_path)
            print(f"  ✅ Nuovo best model salvato (mIoU={best_miou:.4f}) → {ckpt_path}")

    print("\n✅ Training completato!")
    print(f"   Best mIoU: {best_miou:.4f}")
    print(f"   Checkpoint: {os.path.join(Config.ROOT_DIR, 'best_model.pth')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train satellite segmentation model on Kaggle"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Esegui solo 1 batch per testare che tutto funzioni"
    )
    args = parser.parse_args()
    main(args)
