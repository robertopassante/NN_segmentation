import torch
from tqdm import tqdm
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    scaler = torch.amp.GradScaler('cuda')
    
    pbar = tqdm(dataloader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with AMP (Automatic Mixed Precision)
        with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
            logits = model(images)
            loss = criterion(logits, masks)
        
        # Backward and optimize using scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    return running_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader, criterion, device, num_classes=2):
    model.eval()
    running_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    
    # Trackers for IoU and Dice
    total_intersection = torch.zeros(num_classes, device=device)
    total_union = torch.zeros(num_classes, device=device)
    total_target = torch.zeros(num_classes, device=device)
    total_pred = torch.zeros(num_classes, device=device)
    
    pbar = tqdm(dataloader, desc="Evaluating")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
            logits = model(images)
            loss = criterion(logits, masks)
            
        running_loss += loss.item()
        
        # Compute accuracy (ignoring class 0)
        preds = torch.argmax(logits, dim=1)
        valid_mask = masks != 0
        correct_pixels += (preds[valid_mask] == masks[valid_mask]).sum().item()
        total_pixels += valid_mask.sum().item()
        
        # Compute IoU and Dice components
        for c in range(num_classes):
            pred_c = (preds == c)
            mask_c = (masks == c)
            total_intersection[c] += (pred_c & mask_c).sum()
            total_union[c] += (pred_c | mask_c).sum()
            total_target[c] += mask_c.sum()
            total_pred[c] += pred_c.sum()
        
    avg_loss = running_loss / len(dataloader)
    accuracy = correct_pixels / total_pixels
    
    # Calculate macro IoU and Dice
    iou_per_class = total_intersection / (total_union + 1e-6)
    dice_per_class = (2 * total_intersection) / (total_pred + total_target + 1e-6)
    
    # Calculate F1 Score explicitly (Precision & Recall based)
    precision_per_class = total_intersection / (total_pred + 1e-6)
    recall_per_class = total_intersection / (total_target + 1e-6)
    f1_per_class = (2 * precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + 1e-6)
    
    # Average only over classes that are present in the dataset/union
    valid_classes = total_union > 0
    mIoU = iou_per_class[valid_classes].mean().item() if valid_classes.any() else 0.0
    mdice = dice_per_class[valid_classes].mean().item() if valid_classes.any() else 0.0
    mf1 = f1_per_class[valid_classes].mean().item() if valid_classes.any() else 0.0
    
    return avg_loss, accuracy, mIoU, mdice, mf1
