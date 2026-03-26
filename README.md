# Satellite Image Segmentation with Segment-Anything Model (SAM)

This repository contains a PyTorch-based pipeline to fine-tune the Segment-Anything Model (SAM) for remote sensing segmentation tasks (e.g., building footprints, land cover, water bodies). 
It also incorporates ISPAMM lab's wavelet-based strategies (using Discrete Wavelet Transforms) to enhance high-frequency spatial features on satellite imagery.

## 📁 Directory Structure
```
project_root/
├── main.py               # Entry point for training and validation
├── config.py             # Hyperparameters and paths
├── requirements.txt      # Python dependencies
├── data/
│   ├── dataset.py        # Custom PyTorch Dataset reading images and masks
│   └── transforms.py     # Augmentations (includes ISPAMM Wavelet strategy)
├── models/
│   ├── sam_wrapper.py    # Loads SAM's ViT image encoder
│   └── classifier.py     # Custom trainable segmentation head
└── utils/
    ├── engine.py         # train_one_epoch and evaluate loops
    └── plots.py          # Functions for visualizing loss and prediction masks
```

## 🚀 Getting Started

### 1. Installation
Install the required packages. It is highly recommended to use a virtual environment or Conda:
```bash
pip install -r requirements.txt
```

### 2. Download SAM Checkpoint
You must download the pre-trained SAM weights (ViT-B) and place them in the project root folder.
Download link: [sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

*Note: If `main.py` cannot find the weights, it will initialize a dummy encoder for dry-run testing.*

### 3. Setup Dataset
Place your chosen satellite dataset (e.g., LoveDA, Inria, Custom) inside `dataset/` following this structure:
```text
dataset/
├── train/
│   ├── images/  (e.g., 001.tif, 002.png)
│   └── masks/   (e.g., 001.tif, 002.png - 1 channel grayscale labels)
└── val/
    ├── images/
    └── masks/
```

### 4. Training
Once your data and weights are ready, adjust any parameters in `config.py` (like batch size, learning rate, wavelet usage) and run:
```bash
python main.py
```

To quickly verify if the pipeline compiles without waiting for a full epoch, run:
```bash
python main.py --dry-run
```

## ✨ Features Supported
- **Foundation Model Architecture:** Backed by Meta's Segment-Anything Model.
- **Wavelet Enhancement:** Toggled via `config.py`, applies Haar Discrete Wavelet Transform fusion to images to enhance texture and boundary edges.
- **Modular Design:** Easy to swap datasets or augmentations.
