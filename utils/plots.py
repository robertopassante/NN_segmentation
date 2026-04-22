import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import torch
import numpy as np
from config import Config

# ==============================================================================
# Costanti OpenEarthMap
# ==============================================================================
# Mappa indice → nome classe
OEM_CLASS_NAMES = {
    0: "Background",
    1: "Bareland",
    2: "Rangeland",
    3: "Developed",
    4: "Road",
    5: "Tree",
    6: "Water",
    7: "Agriculture",
    8: "Building",
}

# Colormap custom per OpenEarthMap (BGR→RGB normalizzato in [0,1])
# Palette officiale OpenEarthMap: https://open-earth-map.org
OEM_COLORS = np.array([
    [0,   0,   0  ],   # 0: Background   — nero
    [128, 96,  0  ],   # 1: Bareland     — marrone scuro
    [152, 251, 152],   # 2: Rangeland    — verde chiaro
    [220, 220, 220],   # 3: Developed    — grigio chiaro
    [169, 169, 169],   # 4: Road         — grigio scuro
    [34,  139, 34 ],   # 5: Tree         — verde foresta
    [65,  105, 225],   # 6: Water        — blu royal
    [255, 215, 0  ],   # 7: Agriculture  — giallo oro
    [220, 20,  60 ],   # 8: Building     — rosso cremisi
], dtype=np.float32) / 255.0


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Converte una mask (H,W) di indici interi in un'immagine RGB (H,W,3)."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.float32)
    for c, color in enumerate(OEM_COLORS):
        rgb[mask == c] = color
    return rgb


def _per_image_metrics(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """
    Calcola IoU e Dice medi per una singola immagine sulle classi presenti.

    Returns:
        (img_iou, img_dice) come float
    """
    classes_present = np.unique(gt)
    ious, dices = [], []
    for c in classes_present:
        gt_c   = (gt   == c)
        pred_c = (pred == c)
        intersection = np.logical_and(gt_c, pred_c).sum()
        union        = np.logical_or(gt_c, pred_c).sum()
        ious.append(intersection  / (union               + 1e-6))
        dices.append(2 * intersection / (gt_c.sum() + pred_c.sum() + 1e-6))
    return (float(np.mean(ious)) if ious else 0.0,
            float(np.mean(dices)) if dices else 0.0)


# ==============================================================================
# Loss curves
# ==============================================================================
def plot_loss_curves(train_losses, val_losses, save_path="loss_curve.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses,   label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


# ==============================================================================
# Prediction grid — 4 righe fisse (una per classe target)
# ==============================================================================
def save_predictions(images, masks, logits, save_dir, epoch, batch_idx,
                     mIoU=None, mDice=None):
    """
    Salva una griglia di predizioni con esattamente 4 righe,
    una per ciascuna delle classi target di Config.OEM_VIZ_CLASSES.

    Ogni riga mostra:
      Col 0: Immagine originale  [classe dominante]
      Col 1: Ground Truth mask   (colorata con palette OEM)
      Col 2: Prediction mask     (colorata) + IoU/Dice per-immagine

    Se una classe non è presente nel batch corrente, la riga mostra
    un placeholder grigio con testo esplicativo.
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── Prepara i dati dal batch ─────────────────────────────────────────────
    preds  = torch.argmax(logits, dim=1).cpu().numpy()           # (B, H, W)
    images_np = images.cpu().numpy().transpose(0, 2, 3, 1)       # (B, H, W, C)
    masks_np  = masks.cpu().numpy()                              # (B, H, W)

    target_classes = Config.OEM_VIZ_CLASSES  # [6, 8, 5, 7]

    # ── Per ogni classe target: trova l'immagine dove è MORE dominante ───────
    # Struttura: class_id → (batch_index, dominant_pct)
    best_per_class: dict[int, tuple[int, float]] = {}

    for j in range(masks_np.shape[0]):
        total_pixels = masks_np[j].size
        for c in target_classes:
            pct = np.sum(masks_np[j] == c) / total_pixels
            # Soglia minima 5% per non prendere immagini quasi vuote per la classe
            if pct > 0.05 and (c not in best_per_class or pct > best_per_class[c][1]):
                best_per_class[c] = (j, pct)

    # ── Costruisce una lista ordinata di (class_id, batch_index o None) ──────
    rows: list[tuple[int, int | None]] = []
    used_indices: set[int] = set()
    for c in target_classes:
        if c in best_per_class:
            idx = best_per_class[c][0]
            if idx not in used_indices:
                rows.append((c, idx))
                used_indices.add(idx)
            else:
                # L'indice è già usato da un'altra classe: cerca il second-best
                fallback = None
                best_pct = 0.0
                for j in range(masks_np.shape[0]):
                    if j in used_indices:
                        continue
                    pct = np.sum(masks_np[j] == c) / masks_np[j].size
                    if pct > 0.05 and pct > best_pct:
                        fallback = j
                        best_pct = pct
                if fallback is not None:
                    rows.append((c, fallback))
                    used_indices.add(fallback)
                else:
                    rows.append((c, None))  # placeholder
        else:
            rows.append((c, None))  # classe assente nel batch

    n_rows = len(rows)  # sempre 4

    # ── Disegna la figura ──────────────────────────────────────────────────
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = [axes]

    # Titolo globale con metriche di epoch
    epoch_info = ""
    if mIoU is not None:
        epoch_info += f"  mIoU: {mIoU:.4f}"
    if mDice is not None:
        epoch_info += f"  mDice: {mDice:.4f}"
    fig.suptitle(f"Epoch {epoch+1} — Batch {batch_idx}{epoch_info}",
                 fontsize=14, fontweight='bold', y=1.01)

    for row_i, (class_id, batch_idx_img) in enumerate(rows):
        ax_img, ax_gt, ax_pred = axes[row_i]
        class_name = OEM_CLASS_NAMES.get(class_id, f"Classe {class_id}")

        if batch_idx_img is None:
            # ── Placeholder: classe non presente nel batch ────────────────
            for ax in (ax_img, ax_gt, ax_pred):
                ax.set_facecolor("#cccccc")
                ax.text(0.5, 0.5, f"'{class_name}'\nnon presente\nin questo batch",
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=11, color='#444444')
                ax.axis('off')
            ax_img.set_title(f"[{class_id}] {class_name}", fontweight='bold')
            continue

        img  = images_np[batch_idx_img]
        gt   = masks_np[batch_idx_img]
        pred = preds[batch_idx_img]

        # ── Estrai solo RGB e De-normalizza immagine (mean/std ImageNet) ──
        if img.shape[-1] == 4:
            img_rgb = img[..., :3]
        else:
            img_rgb = img
            
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        disp_img = np.clip(img_rgb * std + mean, 0, 1)

        # ── Metriche per-immagine ──────────────────────────────────────────
        img_iou, img_dice = _per_image_metrics(gt, pred)
        dominant_pct = np.sum(gt == class_id) / gt.size

        # ── Colonna 0: immagine originale ─────────────────────────────────
        ax_img.imshow(disp_img)
        ax_img.set_title(f"[{class_id}] {class_name}\n({dominant_pct*100:.1f}% dominante)",
                         fontweight='bold')
        ax_img.axis('off')

        # ── Colonna 1: Ground Truth ───────────────────────────────────────
        ax_gt.imshow(_mask_to_rgb(gt))
        ax_gt.set_title("Ground Truth")
        ax_gt.axis('off')

        # ── Colonna 2: Predizione + IoU/Dice ─────────────────────────────
        ax_pred.imshow(_mask_to_rgb(pred))
        ax_pred.set_title(f"Prediction\nIoU: {img_iou:.3f}  |  Dice: {img_dice:.3f}")
        ax_pred.axis('off')

    # ── Legenda delle classi (sotto la figura) ────────────────────────────
    legend_patches = [
        mpatches.Patch(color=OEM_COLORS[c], label=f"[{c}] {OEM_CLASS_NAMES[c]}")
        for c in range(1, 9)
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=4,
               fontsize=9, title="Classi OpenEarthMap", title_fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    out_path = os.path.join(save_dir, f"epoch_{epoch+1:03d}_batch_{batch_idx}.png")
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  [VIZ] Salvato → {out_path}")
