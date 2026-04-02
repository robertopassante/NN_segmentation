import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from models.lightweight_unet import LightweightUNet
from config import Config

def generate_pseudo_labels(image_path, sam_checkpoint, unet_checkpoint, output_path):
    print("Loading SAM (Segment Anything Model)...")
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    sam.to(Config.DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    print("Loading Trained Swin-Unet...")
    unet = LightweightUNet(num_classes=Config.NUM_CLASSES, encoder_name=Config.ENCODER_NAME, use_satellite_weights=False) # We don't need to load RSP weights here, we load our fine-tuned weights!
    unet.load_state_dict(torch.load(unet_checkpoint, map_location=Config.DEVICE))
    unet.to(Config.DEVICE)
    unet.eval()
    
    # Load and process image
    print(f"Processing Unlabelled Image: {image_path}")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_shape = image.shape[:2]
    
    # Resize for UNet
    img_resized = cv2.resize(image, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
    img_tensor = torch.from_numpy(img_resized.transpose(2,0,1)).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(Config.DEVICE)
    
    print("Generating U-Net Semantic Coarse Prediction...")
    with torch.no_grad():
        logits = unet(img_tensor)
        unet_pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
        # Resize U-Net prediction back to original size
        unet_pred_resized = cv2.resize(unet_pred.astype(np.uint8), (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
        
    print("Generating SAM boundary masks (Zero-Shot)...")
    sam_result = mask_generator.generate(image)
    
    print("Fusing U-Net semantics with SAM crisp boundaries...")
    final_pseudo_label = np.zeros(orig_shape, dtype=np.uint8)
    
    # Sort masks by area descending to paint large backgrounds first, then small objects overlaying them
    sam_result = sorted(sam_result, key=(lambda x: x['area']), reverse=True)
    
    for mask_data in sam_result:
        mask = mask_data['segmentation'] # boolean array
        # Get UNet predictions inside this SAM mask
        masked_preds = unet_pred_resized[mask]
        if len(masked_preds) == 0:
            continue
        # Find majority semantic class inside this crisp SAM polygon
        majority_class = np.bincount(masked_preds).argmax()
        # Paint the final pseudo label
        final_pseudo_label[mask] = majority_class
        
    # Save the pseudo label
    cv2.imwrite(output_path, final_pseudo_label)
    print(f"Saved highly-refined pseudo-label to {output_path}")
    
    # Save a visual comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(image)
    axes[0].set_title("Original Unlabelled Image")
    axes[0].axis('off')
    
    axes[1].imshow(unet_pred_resized, cmap='tab10', vmin=0, vmax=Config.NUM_CLASSES-1, interpolation='nearest')
    axes[1].set_title("U-Net Coarse Prediction\n(Good Classes, Blurry Edges)")
    axes[1].axis('off')
    
    axes[2].imshow(final_pseudo_label, cmap='tab10', vmin=0, vmax=Config.NUM_CLASSES-1, interpolation='nearest')
    axes[2].set_title("SAM-Refined Pseudo-Label\n(Perfect Edges)")
    axes[2].axis('off')
    
    vis_path = output_path.replace(".png", "_vis.png")
    plt.savefig(vis_path)
    plt.close()
    print(f"Saved visual comparison to {vis_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Pseudo-Labels using SAM and trained Swin-Unet")
    parser.add_argument("--image", type=str, required=True, help="Path to unlabelled image (e.g. from internet or test set)")
    parser.add_argument("--unet_ckpt", type=str, default="best_model.pth", help="Your trained weights")
    parser.add_argument("--sam_ckpt", type=str, default="sam_vit_b_01ec64.pth", help="Downloaded SAM weights")
    parser.add_argument("--output", type=str, default="pseudo_label.png", help="Output mask path")
    args = parser.parse_args()
    
    if not os.path.exists(args.sam_ckpt):
        print(f"SAM checkpoint {args.sam_ckpt} not found. Please download it via wget or gdown.")
    elif not os.path.exists(args.unet_ckpt):
        print(f"U-Net checkpoint {args.unet_ckpt} not found! Please train the model first.")
    else:
        generate_pseudo_labels(args.image, args.sam_ckpt, args.unet_ckpt, args.output)
