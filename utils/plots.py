import matplotlib.pyplot as plt
import os
import torch
import numpy as np

def plot_loss_curves(train_losses, val_losses, save_path="loss_curve.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def save_predictions(images, masks, logits, save_dir, epoch, batch_idx, mIoU=None, mDice=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    images = images.cpu().numpy().transpose(0, 2, 3, 1) # B, H, W, C
    masks = masks.cpu().numpy()
    
    # Seleziona 4 immagini dove ogni categoria e' DOMINANTE (la maggioranza dei pixel)
    # LandCover.ai classi: 0=Background, 1=Edifici, 2=Boschi, 3=Acqua, 4=Strade
    target_classes = [1, 2, 3, 4]  # Edifici, Boschi, Acqua, Strade
    
    # Per ogni immagine, calcola quale classe non-background ha piu' pixel
    # Per ogni classe target, tieni l'immagine dove quella classe copre la % piu' alta
    best_per_class = {}  # class -> (image_index, pixel_percentage)
    for j in range(masks.shape[0]):
        total_pixels = masks[j].size
        for c in target_classes:
            count = np.sum(masks[j] == c)
            pct = count / total_pixels
            if pct > 0.05:  # almeno 5% dei pixel per contare
                if c not in best_per_class or pct > best_per_class[c][1]:
                    best_per_class[c] = (j, pct)
    
    # Pesca 1 immagine per categoria, quella dove la classe domina di piu'
    best_indices = []
    used = set()
    for c in target_classes:
        if c in best_per_class:
            idx = best_per_class[c][0]
            if idx not in used:
                best_indices.append(idx)
                used.add(idx)
    
    # Se qualche categoria mancava nel batch, riempi con immagini random non usate
    if len(best_indices) < 4:
        remaining = [j for j in range(masks.shape[0]) if j not in used]
        np.random.shuffle(remaining)
        for idx in remaining:
            if len(best_indices) >= 4:
                break
            best_indices.append(idx)
    
    batch_size = len(best_indices)
    
    fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5*batch_size))
    
    if batch_size == 1:
        axes = [axes]
        
    for i in range(batch_size):
        # Original Image
        ax_img = axes[i][0]
        # De-normalize mathematically to show the true satellite photo
        disp_img = images[best_indices[i]]
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        disp_img = disp_img * std + mean
        disp_img = np.clip(disp_img, 0, 1)
        
        ax_img.imshow(disp_img)
        ax_img.set_title("Input Image")
        ax_img.axis('off')
        
        # Ground Truth Mask
        ax_gt = axes[i][1]
        ax_gt.imshow(masks[best_indices[i]], cmap='tab10', vmin=0, vmax=7, interpolation='nearest')
        ax_gt.set_title("Ground Truth Mask")
        ax_gt.axis('off')
        
        # Prediction
        ax_pred = axes[i][2]
        ax_pred.imshow(preds[best_indices[i]], cmap='tab10', vmin=0, vmax=7, interpolation='nearest')
        title_str = "Model Prediction"
        if mIoU is not None and mDice is not None:
            title_str += f"\nmIoU: {mIoU:.4f} | Dice: {mDice:.4f}"
        ax_pred.set_title(title_str)
        ax_pred.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch}_batch_{batch_idx}.png"))
    plt.close()
