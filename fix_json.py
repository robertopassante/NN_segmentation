import json

with open('Kaggle_Launcher.ipynb', 'r', encoding='utf-8') as f:
    text = f.read()

# Fix the invalid JSON text exactly as it appears
text = text.replace(
    "\"UNLABELED_DIR = '/kaggle/input/global-land-cover-mapping-openearthmap/images/test',",
    "\"UNLABELED_DIR = '/kaggle/input/global-land-cover-mapping-openearthmap/images/test'\\n\",\n"
)

with open('Kaggle_Launcher.ipynb', 'w', encoding='utf-8') as f:
    f.write(text)

# Now we can safely load and append
with open('Kaggle_Launcher.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Ensure the tuple bug inside the code is fixed if it somehow survived
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'UNLABELED_DIR =' in ''.join(cell['source']):
        for i, line in enumerate(cell['source']):
            if 'UNLABELED_DIR =' in line and line.endswith(',\n'):
                cell['source'][i] = line.replace(',\n', '\n')

# Add the Merging Phase (Fase 6)
nb['cells'].append({
   'cell_type': 'markdown',
   'metadata': {},
   'source': [
    '## ⑥ Fase 6: Fusione Dataset (Originale + Pseudo-Labels)\n',
    'Uniamo il dataset di partenza con le nuove maschere per creare un dataset combinato ed evitare il Catastrophic Forgetting.'
   ]
})

nb['cells'].append({
   'cell_type': 'code',
   'execution_count': None,
   'metadata': {},
   'outputs': [],
   'source': [
    '%%bash\n',
    'COMBINED_DIR="/kaggle/working/combined_dataset"\n',
    'mkdir -p $COMBINED_DIR/images/train\n',
    'mkdir -p $COMBINED_DIR/label/train\n',
    'mkdir -p $COMBINED_DIR/images/val\n',
    'mkdir -p $COMBINED_DIR/label/val\n',
    '\n',
    'echo "1. Copia del dataset ORIGINALE in corso..."\n',
    'ORIGINAL_DIR="/kaggle/input/global-land-cover-mapping-openearthmap"\n',
    'cp -r $ORIGINAL_DIR/images/train/* $COMBINED_DIR/images/train/ 2>/dev/null || true\n',
    'cp -r $ORIGINAL_DIR/label/train/* $COMBINED_DIR/label/train/ 2>/dev/null || true\n',
    'cp -r $ORIGINAL_DIR/images/val/* $COMBINED_DIR/images/val/ 2>/dev/null || true\n',
    'cp -r $ORIGINAL_DIR/label/val/* $COMBINED_DIR/label/val/ 2>/dev/null || true\n',
    '\n',
    'echo "2. Copia delle PSEUDO-LABELS in corso..."\n',
    'PSEUDO_DIR="/kaggle/working/pseudo_dataset"\n',
    'cp -r $PSEUDO_DIR/images/* $COMBINED_DIR/images/train/ 2>/dev/null || true\n',
    'cp -r $PSEUDO_DIR/labels/* $COMBINED_DIR/label/train/ 2>/dev/null || cp -r $PSEUDO_DIR/label/* $COMBINED_DIR/label/train/ 2>/dev/null || true\n',
    '\n',
    'echo -e "\\n✅ FUSIONE COMPLETATA! Il nuovo dataset gigante e\' pronto in: $COMBINED_DIR"\n'
   ]
})

# Add the Advanced Training Phase (Fase 7)
nb['cells'].append({
   'cell_type': 'markdown',
   'metadata': {},
   'source': [
    '## ⑦ Fase 7: Training Avanzato (Fine-Tuning)\n',
    'Facciamo ripartire il training usando il nuovo dataset combinato e caricando i pesi pre-addestrati del modello precedente.'
   ]
})

nb['cells'].append({
   'cell_type': 'code',
   'execution_count': None,
   'metadata': {},
   'outputs': [],
   'source': [
    'import os\n',
    '%cd /kaggle/working/NN_segmentation\n',
    '\n',
    'print("Inizio del Secondo Training (Fine-Tuning su dataset combinato)...")\n',
    '\n',
    '# MODIFICA IMPORTANTE:\n',
    '# Se hai caricato best_model.pth come Kaggle Dataset (per saltare il primo training),\n',
    '# modifica "--resume_from" facendolo puntare a quel path, ad esempio:\n',
    '# --resume_from /kaggle/input/my-pretrained-satellite-models/best_model.pth\n',
    '\n',
    '!python main_kaggle.py \\\n',
    '  --data_dir /kaggle/working/combined_dataset \\\n',
    '  --resume_from /kaggle/working/NN_segmentation/best_model.pth\n'
   ]
})

with open('Kaggle_Launcher.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
