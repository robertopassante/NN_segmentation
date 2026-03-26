import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(images)
        loss = criterion(logits, masks)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
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
        
        logits = model(images)
        loss = criterion(logits, masks)
        running_loss += loss.item()
        
        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        correct_pixels += (preds == masks).sum().item()
        total_pixels += torch.numel(masks)
        
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
    
    # Average only over classes that are present in the dataset/union
    valid_classes = total_union > 0
    mIoU = iou_per_class[valid_classes].mean().item() if valid_classes.any() else 0.0
    mdice = dice_per_class[valid_classes].mean().item() if valid_classes.any() else 0.0
    
    return avg_loss, accuracy, mIoU, mdice
