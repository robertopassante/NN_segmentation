"""
prepare_dataset_kaggle.py — Selezione intelligente del subset per Kaggle

Versione Kaggle di data/prepare_dataset.py.
Legge le immagini direttamente da /kaggle/input/ senza torchgeo,
visto che la versione Kaggle del dataset ha una struttura piatta
(images/train/*.tif, label/train/*.tif) invece delle subcartelle per regione.

[ESEGUIRE SU KAGGLE nella cella ④ del Kaggle_Launcher.ipynb]

Output:
  /kaggle/working/NN_segmentation/dataset/oem_train_indices.json
  /kaggle/working/NN_segmentation/dataset/oem_val_indices.json

Gli indici salvati fanno riferimento alla posizione nella lista di file
letta da train.txt / val.txt del dataset Kaggle.
"""

import os
import json
import sys
import numpy as np
import rasterio
from tqdm import tqdm

# Aggiunge la root del progetto al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_kaggle import ConfigKaggle as Config


# ==============================================================================
# Costanti OpenEarthMap (identiche a prepare_dataset.py)
# ==============================================================================
OEM_CLASSES = {
    1: "Bareland",
    2: "Rangeland",
    3: "Developed",
    4: "Road",
    5: "Tree",
    6: "Water",
    7: "Agriculture",
    8: "Building",
}

DOMINANT_THRESHOLD  = Config.DOMINANT_CLASS_THRESHOLD   # 0.40
MAX_TRAIN_PER_CLASS = Config.SAMPLES_PER_CLASS_TRAIN    # 150
MAX_VAL_PER_CLASS   = Config.SAMPLES_PER_CLASS_VAL      # 40


# ==============================================================================
# Core: analisi di un singolo split
# ==============================================================================
def analyze_and_select(split: str, max_per_class: int, output_path: str) -> list:
    """
    Scansiona tutte le immagini di un split e seleziona quelle dominanti.

    Args:
        split: "train" o "val"
        max_per_class: massimo numero di immagini per classe
        output_path: percorso del file JSON di output

    Returns:
        Lista di dict con i metadati dei campioni selezionati
    """
    print(f"\n{'='*65}")
    print(f"  Analisi split: {split.upper()} (versione Kaggle)")
    print(f"  Soglia dominanza  : {DOMINANT_THRESHOLD*100:.0f}%")
    print(f"  Max per classe    : {max_per_class}")
    print(f"{'='*65}")

    # ── Leggi lista file dal txt ufficiale del dataset ────────────────────
    split_txt   = os.path.join(Config.KAGGLE_INPUT_DIR, f"{split}.txt")
    images_dir  = os.path.join(Config.IMAGES_DIR, split)
    labels_dir  = os.path.join(Config.LABELS_DIR, split)

    if not os.path.exists(split_txt):
        print(f"[ERRORE] File split non trovato: {split_txt}")
        print("         Assicurati di aver allegato il dataset Kaggle corretto.")
        return []

    with open(split_txt, "r") as f:
        all_files = [line.strip() for line in f if line.strip()]

    n_total = len(all_files)
    print(f"Immagini totali nel split '{split}': {n_total}\n")

    # ── Contatori e raccoglitori ───────────────────────────────────────────
    class_counts  = {c: 0 for c in OEM_CLASSES}
    selected      = []
    n_no_dominant = 0
    n_class_full  = 0

    pbar = tqdm(enumerate(all_files), total=n_total,
                desc=f"Scansione {split}", unit="img")

    for idx, fname in pbar:

        # Early stop: tutte le classi sono piene
        if all(class_counts[c] >= max_per_class for c in OEM_CLASSES):
            tqdm.write(f"\n✅ Tutte le classi hanno raggiunto {max_per_class}. Stop anticipato a idx={idx}.")
            break

        # ── Leggi la mask (GeoTIFF singolo canale) ────────────────────────
        mask_path = os.path.join(labels_dir, fname)
        if not os.path.exists(mask_path):
            n_no_dominant += 1
            continue

        try:
            with rasterio.open(mask_path) as src:
                mask = src.read(1).astype(np.uint8)     # (H, W)
        except Exception as e:
            tqdm.write(f"[WARN] Impossibile leggere {mask_path}: {e}")
            n_no_dominant += 1
            continue

        total_pixels = mask.size

        # ── Trova la classe dominante ──────────────────────────────────────
        dominant_class = None
        dominant_pct   = 0.0

        for c in OEM_CLASSES:
            pct = np.sum(mask == c) / total_pixels
            if pct >= DOMINANT_THRESHOLD and pct > dominant_pct:
                dominant_class = c
                dominant_pct   = pct

        # ── Filtra e accumula ──────────────────────────────────────────────
        if dominant_class is None:
            n_no_dominant += 1
            continue

        if class_counts[dominant_class] >= max_per_class:
            n_class_full += 1
            continue

        class_counts[dominant_class] += 1
        selected.append({
            "idx":            idx,          # posizione in all_files / split.txt
            "filename":       fname,        # nome file (es. aachen_1.tif)
            "dominant_class": dominant_class,
            "class_name":     OEM_CLASSES[dominant_class],
            "dominant_pct":   round(float(dominant_pct), 4),
        })

        # Aggiorna barra
        pbar.set_postfix({
            "W":  class_counts[6],
            "B":  class_counts[8],
            "T":  class_counts[5],
            "Ag": class_counts[7],
        })

    # ── Riepilogo ──────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  Selezionate    : {len(selected):>5} immagini")
    print(f"  Senza dominante: {n_no_dominant:>5} scartate")
    print(f"  Classe piena   : {n_class_full:>5} scartate")
    print(f"  Per classe:")
    for c, name in OEM_CLASSES.items():
        print(f"    [{c}] {name:<12}: {class_counts[c]:>3}/{max_per_class}")
    print(f"{'─'*50}")

    # ── Salva JSON ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    payload = {
        "split":          split,
        "source":         "kaggle",         # marker per distinguere da versione Colab
        "threshold":      DOMINANT_THRESHOLD,
        "max_per_class":  max_per_class,
        "total_selected": len(selected),
        "class_counts":   {str(k): v for k, v in class_counts.items()},
        "class_names":    {str(k): v for k, v in OEM_CLASSES.items()},
        "samples":        selected,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n💾 Indici salvati → {output_path}")
    return selected


# ==============================================================================
# Entry point
# ==============================================================================
if __name__ == "__main__":
    os.makedirs(Config.DATA_DIR, exist_ok=True)

    train_json = os.path.join(Config.DATA_DIR, "oem_train_indices.json")
    val_json   = os.path.join(Config.DATA_DIR, "oem_val_indices.json")

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   OpenEarthMap (Kaggle) — Preparazione Smart Subset        ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"Input dir : {Config.KAGGLE_INPUT_DIR}")
    print(f"Output dir: {Config.DATA_DIR}")
    print(f"Soglia    : {DOMINANT_THRESHOLD*100:.0f}% pixel dominanti")
    print(f"Max train : {MAX_TRAIN_PER_CLASS} img/classe → ~{MAX_TRAIN_PER_CLASS * len(OEM_CLASSES)} tot")
    print(f"Max val   : {MAX_VAL_PER_CLASS} img/classe → ~{MAX_VAL_PER_CLASS * len(OEM_CLASSES)} tot")

    analyze_and_select("train", MAX_TRAIN_PER_CLASS, train_json)
    analyze_and_select("val",   MAX_VAL_PER_CLASS,   val_json)

    print("\n✅ Preparazione Kaggle completata!")
    print(f"   Train → {train_json}")
    print(f"   Val   → {val_json}")
    print("\n💡 Copia questi JSON su /kaggle/working/ e salvali come Dataset Kaggle")
    print("   'oem-indices' per riutilizzarli nelle sessioni future.")
