import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import argparse

from config import Config
from data.dataset import SatelliteSegmentationDataset
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
    if Config.DATASET_NAME.lower() != "openearthmap":
        return None  # Pools limitati solo a OEM per la compatibilità retroattiva
        
    print("\n[VIZ] Scansione validation set per creare i batch di visualizzazione...")
    pools = {c: [] for c in target_classes}
    indices_to_check = list(range(len(dataset)))
    
    for idx in tqdm(indices_to_check, desc="Scanning val masks"):
        _, mask = dataset[idx]
        mask_np = mask.numpy() if hasattr(mask, 'numpy') else mask
        for c in target_classes:
            if (mask_np == c).mean() > 0.05:
                pools[c].append(idx)
                
    for c in target_classes:
        if len(pools[c]) == 0:
            print(f"[WARN] Classe {c} non trovata nel validation set (>5% px). Fallback a random.")
            pools[c] = list(range(len(dataset)))
    return pools


def main(args):
    print("Initializing Neural Network Project: Satellite Image Segmentation")
    print(f"Using Device: {Config.DEVICE}")
    print(f"Wavelet Augmentation (ISPAMM Strategy): {'Enabled' if Config.USE_WAVELET_AUGMENTATION else 'Disabled'}")
    
    # 1. Dataset & Dataloaders
    train_transform = get_train_transforms(Config.IMAGE_SIZE, use_wavelet=Config.USE_WAVELET_AUGMENTATION)
    val_transform = get_val_transforms(Config.IMAGE_SIZE, use_wavelet=False) # Skip wavelet on val for pure inference evaluation
    
    train_dataset = SatelliteSegmentationDataset(data_dir=Config.DATA_DIR, transform=train_transform, split="train")
    val_dataset = SatelliteSegmentationDataset(data_dir=Config.DATA_DIR, transform=val_transform, split="val")
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)
    
    # ── 1.5. Prepara i pool per la visualizzazione dinamica ──────────────
    viz_pools = prepare_visualization_pools(val_dataset, Config.OEM_VIZ_CLASSES) if Config.DATASET_NAME.lower() == "openearthmap" else None

    # ── 2. Modello ───────────────────────────────────────────────────────
    # Initialize Lightweight Swin U-Net backbone
    model = LightweightUNet(
        num_classes=Config.NUM_CLASSES,
        encoder_name=Config.ENCODER_NAME
    )
    
    # Se abbiamo più di 1 GPU, parallelizziamo
    if torch.cuda.device_count() > 1:
        print(f"\n🚀 Trovate {torch.cuda.device_count()} GPU! Attivazione DataParallel...")
        model = torch.nn.DataParallel(model)
        
    model = model.to(Config.DEVICE)
    
    # 3. Loss, Optimizer, and Scheduler
    print("Setting up loss and optimizer...")
    
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    for param in base_model.model.encoder.parameters():
        param.requires_grad = False
        
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.LEARNING_RATE)
    import segmentation_models_pytorch as smp_module
    class CombinedLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.ce = nn.CrossEntropyLoss(ignore_index=0)
            self.dice = smp_module.losses.DiceLoss(mode='multiclass', classes=[1,2,3,4,5,6,7,8])
        def forward(self, preds, targets):
            return self.ce(preds, targets) + self.dice(preds, targets)
            
    criterion = CombinedLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)
    
    train_losses = []
    val_losses = []
    
    print("Starting Training Loop...")
    for epoch in range(Config.NUM_EPOCHS):
        # Unfreeze backbone after 3 epochs
        if epoch == 3:
            print("Unfreezing backbone for fine-tuning...")
            base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            for param in base_model.model.encoder.parameters():
                param.requires_grad = True
            # Reinizializza l'ottimizzatore per includere il backbone con un learning rate bassissimo
            optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE / 10)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS - 3)
        print(f"\n--- Epoch {epoch+1}/{Config.NUM_EPOCHS} ---")
        
        # Stop dryrun after 1 batch
        if args.dry_run:
            print("[Dry Run] Skipped full epoch calculation.")
            break
            
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        scheduler.step()
        val_loss, val_acc, val_miou, val_dice = evaluate(model, val_loader, criterion, Config.DEVICE, num_classes=Config.NUM_CLASSES)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc*100:.2f}% | mIoU: {val_miou:.4f} | Dice: {val_dice:.4f}")
        
        # Save plots
        plot_loss_curves(train_losses, val_losses)
        
        # ── Visualizzazione predizioni per classe ────────────────────────
        model.eval()
        with torch.no_grad():
            try:
                if Config.DATASET_NAME.lower() == "openearthmap" and viz_pools:
                    # Seleziona 4 indici casuali (uno per ogni target class) dai pool pre-calcolati
                    viz_indices = [random.choice(viz_pools[c]) for c in Config.OEM_VIZ_CLASSES]
                    imgs_list, msks_list = [], []
                    for idx in viz_indices:
                        img, msk = val_dataset[idx]
                        imgs_list.append(img)
                        msks_list.append(msk)
                    val_imgs = torch.stack(imgs_list).to(Config.DEVICE)
                    val_msks = torch.stack(msks_list).to(Config.DEVICE)
                else:
                    val_imgs, val_msks = next(iter(val_loader))
                    val_imgs = val_imgs[:4].to(Config.DEVICE)
                    val_msks = val_msks[:4].to(Config.DEVICE)
                    
                logits = model(val_imgs)
                save_predictions(
                    val_imgs, val_msks, logits,
                    save_dir=os.path.join(Config.ROOT_DIR, "output_samples"),
                    epoch=epoch, batch_idx=0,
                    mIoU=val_miou, mDice=val_dice
                )
            except Exception as e:
                print(f"[WARN] Skip visualizzazione: {e}")
                
        # Save model
        base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        torch.save(base_model.state_dict(), "best_model.pth")
        
    print("Training process finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAM for Satellite Segmentation")
    parser.add_argument("--dry-run", action="store_true", help="Run a quick test without actual training loop")
    args = parser.parse_args()
    
    # Small test if data folders don't have images
    main(args)
