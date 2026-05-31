# Hybrid CNN+BiLSTM Framework for Multiclass Oral Cancer Detection

A deep learning framework that fuses **histopathology image features** (EfficientNet-B3) with **longitudinal clinical sequences** (BiLSTM) for four-class oral cancer classification — achieving **100% accuracy** across all 5-fold cross-validation folds.

---

## 🏆 Key Results

| Model | Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|
| Random Forest | 99.67% | 99.67% | 1.0000 |
| SVM (RBF) | 99.17% | 99.17% | 0.9999 |
| DenseNet121 + Clinical | 96.17% | 96.13% | 0.9972 |
| EfficientNet-B0 + Clinical | 95.83% | 95.75% | 0.9979 |
| MobileNetV3 + Clinical | 95.33% | 95.28% | 0.9940 |
| VGG16 + Clinical | 95.16% | 95.05% | 0.9913 |
| ResNet50 + Clinical | 94.33% | 94.28% | 0.9937 |
| **CNN+BiLSTM Hybrid (Proposed)** | **100.00%** | **100.00%** | **1.0000** |

> All results from **5-fold stratified cross-validation** · 50 epochs · CPU only

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [Qualitative Results](#qualitative-results)
- [License](#license)

---

## Overview

Oral cancer is one of the most prevalent and life-threatening malignancies worldwide, with late-stage diagnosis significantly worsening patient prognosis. This project proposes a **multimodal hybrid deep learning framework** that simultaneously processes:

- 🖼️ **Histopathology images** — spatial tissue morphology captured by a frozen EfficientNet-B3 CNN encoder
- 📊 **Clinical feature sequences** — longitudinal patient records (age, tobacco exposure, lesion characteristics, etc.) modelled by a Bidirectional LSTM

The fusion of both modalities yields a **clinically meaningful and statistically superior** diagnostic system compared to any single-modality approach.

**Target classes:**
| Class | Description |
|---|---|
| `Leukoplakia_Dysplasia` | Leukoplakia with cellular dysplasia (precancerous) |
| `Leukoplakia_NoDysplasia` | Leukoplakia without dysplasia (benign lesion) |
| `Normal` | Healthy oral tissue |
| `OSCC` | Oral Squamous Cell Carcinoma (malignant) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              CNN+BiLSTM Hybrid Framework                │
├───────────────────────┬─────────────────────────────────┤
│     CNN Encoder       │      BiLSTM Encoder             │
│  (EfficientNet-B3)    │   (2-layer Bidirectional LSTM)  │
│  Frozen backbone      │   hidden=128, output=256-dim    │
│  512-dim embedding    │   LayerNorm on output           │
└──────────┬────────────┴────────────┬────────────────────┘
           │                         │
           └────────┬────────────────┘
                    ▼
         Concatenation [512 + 256 = 768-dim]
                    │
         ┌──────────▼──────────┐
         │   FusionClassifier  │
         │  Dense(256) → BN    │
         │  → ReLU → Drop(0.4) │
         │  → Dense(128)       │
         │  → ReLU → Drop(0.2) │
         │  → Dense(4)         │
         └──────────┬──────────┘
                    ▼
         4-class Softmax Output
   (LD | LND | Normal | OSCC)
```

**Training details:**
- Optimiser: AdamW (`lr=1e-4`, `weight_decay=1e-5`)
- LR Schedule: Linear warmup (5 ep) → Cosine Annealing
- Loss: Class-weighted Cross-Entropy
- Gradient clipping: `max_norm=1.0`
- Early stopping patience: 15 epochs
- Best checkpoint saved and reloaded before evaluation

---

## Dataset

- **600 patients** — 150 per class (perfectly balanced)
- **Images:** Oral histopathology patches from the [Mendeley Oral Cancer Dataset](https://data.mendeley.com/)
- **Clinical features:** 13 patient-level attributes per visit (up to 10 visits/patient)

| Feature | Description |
|---|---|
| Age | Patient age |
| Tobacco use | Smoking/chewing tobacco history |
| Alcohol consumption | Alcohol use frequency |
| Oral lesion presence | Binary lesion indicator |
| Lesion site | Anatomical location |
| Lesion size | Measured dimensions |
| Pain score | Self-reported pain level |
| Oral hygiene | Hygiene rating |
| + 5 more | Additional clinical indicators |

**Image preprocessing:**
- Resize to `224 × 224`
- Normalise: `μ=[0.485, 0.456, 0.406]`, `σ=[0.229, 0.224, 0.225]`
- Augmentation: random flip, rotation ±15°, colour jitter, random crop

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/Hybrid-CNNBiLSTM-Oral-Cancer-Detection.git
cd Hybrid-CNNBiLSTM-Oral-Cancer-Detection
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**Requirements:**
```
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
Pillow>=10.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
```

---

## Project Structure

```
Hybrid-CNNBiLSTM-Oral-Cancer-Detection/
│
├── main.py                    # OralCancerHybridModel, BiLSTM, FusionClassifier
├── run_single_model.py        # Train any single model (5-fold CV)
├── baseline_comparison.py     # CNN baseline models & dataset classes
├── prepare_and_run.py         # Data preparation, config, image collection
├── generate_comparison.py     # Final comparison table generator
├── generate_qualitative.py    # Qualitative result figure generator
├── generate_results.py        # Result aggregation utilities
├── data_prep.py               # Data preprocessing utilities
├── requirements.txt           # Python dependencies
│
├── data/
│   └── clinical/
│       └── merged_clinical.csv   # Patient clinical feature records
│
└── results/
    ├── *_results.json            # Per-model 5-fold CV metrics
    ├── *_confusion_matrix.png    # Confusion matrices
    ├── *_report.txt              # Classification reports
    ├── comparison_table.csv      # Full model comparison
    ├── comparison_table.tex      # LaTeX-ready comparison table
    ├── comparison_*.png          # Bar charts and heatmaps
    ├── experiments_and_results.tex   # LaTeX: experiments & results section
    ├── conclusion_and_future_scope.tex  # LaTeX: conclusion & future scope
    └── qualitative/
        ├── qualitative_grid.png           # 4×3 class prediction grid
        ├── qualitative_representative.png # 1 image per class
        └── qualitative_correct_vs_error.png
```

---

## Usage

### Train a single model

```bash
# Classical ML (trains on clinical features only)
python run_single_model.py --model random_forest
python run_single_model.py --model svm

# CNN + Clinical Fusion baselines
python run_single_model.py --model mobilenet_v3
python run_single_model.py --model efficientnet_b0
python run_single_model.py --model densenet121
python run_single_model.py --model resnet50
python run_single_model.py --model vgg16

# Proposed Hybrid CNN+BiLSTM
python run_single_model.py --model hybrid
```

### Run full pipeline (all models sequentially)

```bash
python run_single_model.py --model random_forest
python run_single_model.py --model svm
python run_single_model.py --model mobilenet_v3
python run_single_model.py --model efficientnet_b0
python run_single_model.py --model densenet121
python run_single_model.py --model resnet50
python run_single_model.py --model vgg16
python run_single_model.py --model hybrid
```

### Generate final comparison table

```bash
python generate_comparison.py
```

Outputs: `results/comparison_table.csv`, `results/comparison_table.tex`, and multiple chart PNGs.

### Generate qualitative result figures

```bash
python generate_qualitative.py
```

Outputs 3 figures to `results/qualitative/`.

---

## Results

### Per-fold results — Proposed CNN+BiLSTM Hybrid

| Fold | Accuracy | F1-Score | Precision | AUC-ROC | Best Epoch |
|---|---|---|---|---|---|
| 1 | 100.00% | 100.00% | 100.00% | 1.0000 | 16 |
| 2 | 100.00% | 100.00% | 100.00% | 0.9999 | 13 |
| 3 | 100.00% | 100.00% | 100.00% | 1.0000 | 14 |
| 4 | 100.00% | 100.00% | 100.00% | 1.0000 | 11 |
| 5 | 100.00% | 100.00% | 100.00% | 1.0000 | 14 |
| **Mean ± Std** | **100.00 ± 0.00%** | **100.00 ± 0.00%** | **100.00 ± 0.00%** | **1.0000 ± 0.0000** | — |

### Improvement over baselines

| Baseline | ΔAcc% | ΔF1% | ΔAUC-ROC |
|---|---|---|---|
| Random Forest | +0.33 | +0.33 | 0.0000 |
| SVM (RBF) | +0.83 | +0.83 | +0.0001 |
| MobileNetV3 | +4.67 | +4.72 | +0.0060 |
| EfficientNet-B0 | +4.17 | +4.25 | +0.0021 |
| DenseNet121 | +3.83 | +3.87 | +0.0028 |
| ResNet50 | +5.67 | +5.72 | +0.0063 |
| VGG16 | +4.84 | +4.95 | +0.0087 |

---

## Qualitative Results

The proposed model correctly classifies **all samples** across all four oral pathology classes with zero misclassifications on the validation set.

| Class | Sample Count | Correct | Incorrect |
|---|---|---|---|
| Leukoplakia Dysplasia | 30 | 30 ✅ | 0 |
| Leukoplakia No Dysplasia | 30 | 30 ✅ | 0 |
| Normal | 30 | 30 ✅ | 0 |
| OSCC | 30 | 30 ✅ | 0 |

See `results/qualitative/` for annotated prediction figures.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- **Mendeley Oral Cancer Histopathology Dataset** — image data source
- **PyTorch** and **scikit-learn** — deep learning and ML frameworks
- **EfficientNet-B3** pretrained weights via `torchvision.models`
