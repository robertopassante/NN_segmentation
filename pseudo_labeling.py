import os
import glob
import torch
import numpy as np
import cv2
import pywt
from tqdm import tqdm
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from models.lightweight_unet import LightweightUNet

try:
    from config_kaggle import ConfigKaggle as Config
except ImportError:
    from config import Config

def process_image_for_unet(image_rgb):
    """Prepara l'immagine per la U-Net (resize, eventuale Wavelet e normalizzazione ImageNet)"""
    orig_shape = image_rgb.shape[:2]
    # Resize to UNet input size
    img_resized = cv2.resize(image_rgb, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
    
    if getattr(Config, 'USE_WAVELET_AUGMENTATION', False):
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        LL, (LH, HL, HH) = pywt.dwt2(gray, getattr(Config, 'WAVELET_TYPE', 'haar'))
        edges = np.abs(LH) + np.abs(HL) + np.abs(HH)
        edges = cv2.resize(edges, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
        edges = np.clip(edges, 0, 255).astype(np.uint8)
        img_resized = np.dstack([img_resized, edges])
    
    # Normalizzazione manuale identica a Albumentations A.Normalize
    mean = np.array([0.485, 0.456, 0.406, 0.5]) if img_resized.shape[-1] == 4 else np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225, 0.5]) if img_resized.shape[-1] == 4 else np.array([0.229, 0.224, 0.225])
    
    img_float = img_resized.astype(np.float32) / 255.0
    img_norm = (img_float - mean) / std
    
    img_tensor = torch.from_numpy(img_norm.transpose(2,0,1)).float()
    return img_tensor.unsqueeze(0).to(Config.DEVICE), orig_shape

def generate_pseudo_labels_batch(input_dir, sam_checkpoint, unet_checkpoint, output_dir):
    print("Loading SAM (Segment Anything Model)...")
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    sam.to(Config.DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    print("Loading Trained Swin-Unet...")
    in_channels = 4 if getattr(Config, 'USE_WAVELET_AUGMENTATION', False) else 3
    unet = LightweightUNet(num_classes=Config.NUM_CLASSES, encoder_name=Config.ENCODER_NAME, use_satellite_weights=False, in_channels=in_channels)
    unet.load_state_dict(torch.load(unet_checkpoint, map_location=Config.DEVICE))
    unet.to(Config.DEVICE)
    unet.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Supporta estensioni TIF, PNG, JPG
    image_paths = glob.glob(os.path.join(input_dir, "*.tif")) + glob.glob(os.path.join(input_dir, "*.png")) + glob.glob(os.path.join(input_dir, "*.jpg"))
    print(f"Trovate {len(image_paths)} immagini unlabelled in {input_dir}")
    
    for img_path in tqdm(image_paths, desc="Pseudo-Labeling Batch"):
        filename = os.path.basename(img_path)
        out_mask_path = os.path.join(output_dir, filename.replace(".jpg", ".png").replace(".tif", ".png"))
        out_vis_path = os.path.join(vis_dir, filename.replace(".jpg", ".png").replace(".tif", ".png"))
        
        image = cv2.imread(img_path)
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        img_tensor, orig_shape = process_image_for_unet(image_rgb)
        
        with torch.no_grad():
            logits = unet(img_tensor)
            unet_pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
            unet_pred_resized = cv2.resize(unet_pred.astype(np.uint8), (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
            
        sam_result = mask_generator.generate(image_rgb) # SAM lavora in RGB originale!
        final_pseudo_label = np.zeros(orig_shape, dtype=np.uint8)
        
        # Sort per riempire prima i background larghi, per poi sovrascrivere con oggetti piccoli
        sam_result = sorted(sam_result, key=(lambda x: x['area']), reverse=True)
        
        for mask_data in sam_result:
            mask = mask_data['segmentation'] 
            masked_preds = unet_pred_resized[mask]
            if len(masked_preds) == 0:
                continue
            majority_class = np.bincount(masked_preds).argmax()
            final_pseudo_label[mask] = majority_class
            
        cv2.imwrite(out_mask_path, final_pseudo_label)
        
        # Salvataggio plot comparativo
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(image_rgb)
        axes[0].set_title("Original")
        axes[0].axis('off')
        axes[1].imshow(unet_pred_resized, cmap='tab10', vmin=0, vmax=Config.NUM_CLASSES-1, interpolation='nearest')
        axes[1].set_title("U-Net Coarse (Blocchi)")
        axes[1].axis('off')
        axes[2].imshow(final_pseudo_label, cmap='tab10', vmin=0, vmax=Config.NUM_CLASSES-1, interpolation='nearest')
        axes[2].set_title("SAM Pseudo-Label (Precisione Sub-pixel)")
        axes[2].axis('off')
        plt.savefig(out_vis_path, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Pseudo-Labels per una directory di immagini")
    parser.add_argument("--input_dir", type=str, required=True, help="Cartella con immagini senza maschera")
    parser.add_argument("--output_dir", type=str, required=True, help="Cartella dove salvare le maschere generate")
    parser.add_argument("--unet_ckpt", type=str, default="best_model.pth", help="I tuoi pesi addestrati al 50%")
    parser.add_argument("--sam_ckpt", type=str, default="sam_vit_b_01ec64.pth", help="I pesi SAM scaricati")
    args = parser.parse_args()
    
    if not os.path.exists(args.sam_ckpt):
        print(f"SAM checkpoint {args.sam_ckpt} non trovato.")
    elif not os.path.exists(args.unet_ckpt):
        print(f"U-Net checkpoint {args.unet_ckpt} non trovato.")
    else:
        generate_pseudo_labels_batch(args.input_dir, args.sam_ckpt, args.unet_ckpt, args.output_dir)
