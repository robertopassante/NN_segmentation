"""
prepare_dataset.py — Selezione intelligente del subset OpenEarthMap

Analizza ogni immagine del dataset e seleziona solo quelle in cui una singola
classe è dominante (>= DOMINANT_CLASS_THRESHOLD dei pixel della mask).
Salva gli indici selezionati in file JSON che vengono poi usati da dataset.py.

[ESEGUIRE SU GOOGLE COLAB prima del training, dopo aver scaricato OpenEarthMap]

Flusso:
  1. Carica il dataset raw tramite torchgeo (split='train' e split='val')
  2. Per ogni immagine, analizza la distribuzione dei pixel nella mask
  3. Se una classe non-background ha >= 40% dei pixel → l'immagine è "dominata" da quella classe
  4. Accumula fino a SAMPLES_PER_CLASS_TRAIN/VAL immagini per classe
  5. Salva gli indici in:
       dataset/oem_train_indices.json
       dataset/oem_val_indices.json

OpenEarthMap classi (indici interi nella mask):
  0: Background (no-data)
  1: Bareland
  2: Rangeland
  3: Developed
  4: Road
  5: Tree
  6: Water
  7: Agriculture
  8: Building
"""

import os
import json
import sys
import numpy as np
from tqdm import tqdm

# Aggiunge la root del progetto al path per importare Config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from torchgeo.datasets import OpenEarthMap


# ==============================================================================
# Costanti
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

DOMINANT_THRESHOLD  = Config.DOMINANT_CLASS_THRESHOLD  # 0.40
MAX_TRAIN_PER_CLASS = Config.SAMPLES_PER_CLASS_TRAIN   # 150
MAX_VAL_PER_CLASS   = Config.SAMPLES_PER_CLASS_VAL     # 40


# ==============================================================================
# Core: analisi di un singolo split
# ==============================================================================
def analyze_and_select(split: str, max_per_class: int, output_path: str) -> list:
    """
    Scansiona tutte le immagini di un dato split e seleziona quelle dominanti.

    Args:
        split: "train" o "val"
        max_per_class: massimo numero di immagini da tenere per ogni classe
        output_path: percorso del file JSON di output

    Returns:
        Lista di dict con le informazioni sui campioni selezionati
    """
    print(f"\n{'='*65}")
    print(f"  Analisi split: {split.upper()}")
    print(f"  Soglia dominanza  : {DOMINANT_THRESHOLD*100:.0f}%")
    print(f"  Max per classe    : {max_per_class}")
    print(f"{'='*65}")

    # Carica il dataset raw (senza subset)
    try:
        geo_dataset = OpenEarthMap(
            root=Config.DATA_DIR,
            split=split,
            download=False,
            checksum=False,
        )
    except Exception as e:
        print(f"[ERRORE] Impossibile caricare OpenEarthMap (split={split}): {e}")
        print("Assicurati che il dataset sia scaricato in:", Config.DATA_DIR)
        return []

    n_total = len(geo_dataset)
    print(f"Immagini totali nel split '{split}': {n_total}\n")

    # Contatori e raccoglitori
    class_counts    = {c: 0 for c in OEM_CLASSES}
    selected        = []          # lista di dict con idx + metadati
    n_no_dominant   = 0           # immagini senza classe dominante
    n_class_full    = 0           # immagini scartate perché la classe era piena

    pbar = tqdm(range(n_total), desc=f"Scansione {split}", unit="img")
    for idx in pbar:

        # ── Early stop: tutte le classi sono piene ──────────────────────────
        if all(class_counts[c] >= max_per_class for c in OEM_CLASSES):
            tqdm.write(f"\n✅ Tutte le classi hanno raggiunto {max_per_class}. Stop anticipato a indice {idx}.")
            break

        # ── Carica la mask ───────────────────────────────────────────────────
        sample = geo_dataset[idx]
        mask   = sample["mask"].numpy().squeeze()   # (H, W) con valori 0-8
        total_pixels = mask.size

        # ── Trova la classe dominante (esclude background = 0) ───────────────
        dominant_class = None
        dominant_pct   = 0.0

        for c in OEM_CLASSES:
            pct = np.sum(mask == c) / total_pixels
            if pct >= DOMINANT_THRESHOLD and pct > dominant_pct:
                dominant_class = c
                dominant_pct   = pct

        # ── Filtra e accumula ────────────────────────────────────────────────
        if dominant_class is None:
            n_no_dominant += 1
            continue

        if class_counts[dominant_class] >= max_per_class:
            n_class_full += 1
            continue

        class_counts[dominant_class] += 1
        selected.append({
            "idx":            idx,
            "dominant_class": dominant_class,
            "class_name":     OEM_CLASSES[dominant_class],
            "dominant_pct":   round(float(dominant_pct), 4),
        })

        # Aggiorna barra di avanzamento con conteggio classi chiave
        pbar.set_postfix({
            "W":  class_counts[6],   # Water
            "B":  class_counts[8],   # Building
            "T":  class_counts[5],   # Tree
            "Ag": class_counts[7],   # Agriculture
        })

    # ── Riepilogo ────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  Selezionate    : {len(selected):>5} immagini")
    print(f"  Senza dominante: {n_no_dominant:>5} scartate")
    print(f"  Classe piena   : {n_class_full:>5} scartate")
    print(f"  Per classe:")
    for c, name in OEM_CLASSES.items():
        bar = "█" * class_counts[c] + "░" * (max_per_class - class_counts[c])
        # Stampa solo una barra semplice senza unicode overflow
        print(f"    [{c}] {name:<12}: {class_counts[c]:>3}/{max_per_class}")
    print(f"{'─'*50}")

    # ── Salva JSON ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    payload = {
        "split":           split,
        "threshold":       DOMINANT_THRESHOLD,
        "max_per_class":   max_per_class,
        "total_selected":  len(selected),
        "class_counts":    {str(k): v for k, v in class_counts.items()},
        "class_names":     {str(k): v for k, v in OEM_CLASSES.items()},
        "samples":         selected,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n💾 Indici salvati → {output_path}")
    return selected


# ==============================================================================
# Entry point
# ==============================================================================
if __name__ == "__main__":
    train_json = os.path.join(Config.DATA_DIR, "oem_train_indices.json")
    val_json   = os.path.join(Config.DATA_DIR, "oem_val_indices.json")

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     OpenEarthMap — Preparazione Smart Subset                ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"DATA_DIR : {Config.DATA_DIR}")
    print(f"Soglia   : {DOMINANT_THRESHOLD*100:.0f}% pixel dominanti")
    print(f"Max train: {MAX_TRAIN_PER_CLASS} img/classe → ~{MAX_TRAIN_PER_CLASS * len(OEM_CLASSES)} tot")
    print(f"Max val  : {MAX_VAL_PER_CLASS} img/classe → ~{MAX_VAL_PER_CLASS * len(OEM_CLASSES)} tot")

    analyze_and_select("train", MAX_TRAIN_PER_CLASS, train_json)
    analyze_and_select("val",   MAX_VAL_PER_CLASS,   val_json)

    print("\n✅ Preparazione completata!")
    print(f"   Train → {train_json}")
    print(f"   Val   → {val_json}")
    print("\n➡  Ora puoi avviare il training con main.py")
