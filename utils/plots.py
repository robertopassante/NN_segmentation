import matplotlib.pyplot as plt
import os
import torch

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

def save_predictions(images, masks, logits, save_dir, epoch, batch_idx):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    images = images.cpu().numpy().transpose(0, 2, 3, 1) # B, H, W, C
    masks = masks.cpu().numpy()
    
    # Save up to 4 images from the batch
    batch_size = min(4, images.shape[0])
    
    fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5*batch_size))
    
    if batch_size == 1:
        axes = [axes]
        
    for i in range(batch_size):
        # Original Image
        ax_img = axes[i][0]
        # De-normalize slightly for display if needed
        disp_img = images[i]
        ax_img.imshow(disp_img)
        ax_img.set_title("Input Image")
        ax_img.axis('off')
        
        # Ground Truth Mask
        ax_gt = axes[i][1]
        ax_gt.imshow(masks[i], cmap='gray')
        ax_gt.set_title("Ground Truth Mask")
        ax_gt.axis('off')
        
        # Prediction
        ax_pred = axes[i][2]
        ax_pred.imshow(preds[i], cmap='jet')
        ax_pred.set_title("Model Prediction")
        ax_pred.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch}_batch_{batch_idx}.png"))
    plt.close()
