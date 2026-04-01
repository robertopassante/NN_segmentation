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
    
    # 2. Model Initialization
    # Initialize Lightweight Swin U-Net backbone
    model = LightweightUNet(
        num_classes=Config.NUM_CLASSES,
        encoder_name=Config.ENCODER_NAME
    )
    model = model.to(Config.DEVICE)
    
    # 3. Setup Optimizer and Loss
    # Optimize all parameters of the lightweight model
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    
    print("Starting Training Loop...")
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{Config.NUM_EPOCHS} ---")
        
        # Stop dryrun after 1 batch
        if args.dry_run:
            print("[Dry Run] Skipped full epoch calculation.")
            break
            
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        val_loss, val_acc, val_miou, val_dice = evaluate(model, val_loader, criterion, Config.DEVICE, num_classes=Config.NUM_CLASSES)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc*100:.2f}% | mIoU: {val_miou:.4f} | Dice: {val_dice:.4f}")
        
        # Save plots
        plot_loss_curves(train_losses, val_losses)
        
        # Generate some sample predictions
        model.eval()
        with torch.no_grad():
            try:
                # Grab a batch from validation
                val_imgs, val_msks = next(iter(val_loader))
                val_imgs, val_msks = val_imgs.to(Config.DEVICE), val_msks.to(Config.DEVICE)
                logits = model(val_imgs)
                save_predictions(val_imgs, val_msks, logits, "output_samples", epoch, 0, mIoU=val_miou, mDice=val_dice)
            except Exception as e:
                # Might trigger if val_loader assumes empty data during setup
                print(f"Skipping visualization for now: {e}")
                
        # Save model
        torch.save(model.state_dict(), "best_model.pth")
        
    print("Training process finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAM for Satellite Segmentation")
    parser.add_argument("--dry-run", action="store_true", help="Run a quick test without actual training loop")
    args = parser.parse_args()
    
    # Small test if data folders don't have images
    main(args)
