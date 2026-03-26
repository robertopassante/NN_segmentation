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
        
    avg_loss = running_loss / len(dataloader)
    accuracy = correct_pixels / total_pixels
    
    return avg_loss, accuracy
