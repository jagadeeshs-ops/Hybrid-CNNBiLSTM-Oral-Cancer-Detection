"""
generate_qualitative.py
-----------------------
Generates qualitative result figures for the proposed CNN+BiLSTM Hybrid:
  - 3 sample images per class (LD, LND, Normal, OSCC)
  - Each image annotated with Ground Truth and Predicted label
  - Saves to results/qualitative/
"""

import os, sys, json, random
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from prepare_and_run import build_sequences, assign_images, collect_images, RUN_CONFIG
from baseline_comparison import build_image_transforms
from main import OralCancerHybridModel, OralCancerDataset

SEED      = 42
N_FOLDS   = 5
BATCH     = 8
IMG_SIZE  = 224
DEVICE    = "cpu"
MODELS_DIR = "models"
OUT_DIR    = os.path.join("results", "qualitative")
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

CLASS_NAMES  = ['Leukoplakia_Dysplasia', 'Leukoplakia_NoDysplasia', 'Normal', 'OSCC']
CLASS_LABELS = ['Leukoplakia\nDysplasia', 'Leukoplakia\nNo Dysplasia', 'Normal', 'OSCC']
CLASS_COLORS = ['#E74C3C', '#F39C12', '#27AE60', '#8E44AD']   # red, orange, green, purple

# ── Load data ────────────────────────────────────────────────────────────────
csv_path    = os.path.join("data", "clinical", "merged_clinical.csv")
cdf         = pd.read_csv(csv_path)
sequences, labels, pids, class_names = build_sequences(cdf, seq_len=10)
raw_images  = collect_images()
image_paths = assign_images(raw_images, n_patients=len(labels))

print(f"[Data] sequences={sequences.shape}  labels={labels.shape}")
print(f"[Data] classes={class_names}")
print(f"[Data] {len(raw_images)} images → {len(image_paths)} assigned paths")

# ── Use Fold 5 val set (last fold of 5-fold CV) ───────────────────────────────
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
splits = list(skf.split(image_paths, labels))
tr_idx, vl_idx = splits[-1]   # Fold 5

vl_imgs = [image_paths[i] for i in vl_idx]
vl_seqs = sequences[vl_idx]
vl_lbls = labels[vl_idx]

val_ds     = OralCancerDataset(vl_imgs, vl_seqs, vl_lbls,
                transform=build_image_transforms(False, IMG_SIZE))
val_loader = DataLoader(val_ds, BATCH, shuffle=False, num_workers=0)

# ── Load best hybrid model (Fold 5) ──────────────────────────────────────────
n_features = sequences.shape[2]
cfg = {**RUN_CONFIG, "n_folds": N_FOLDS, "epochs": 50, "batch_size": BATCH,
       "patience": 10, "device": DEVICE, "num_workers": 0, "num_classes": 4}

model = OralCancerHybridModel(n_features, cfg).to(DEVICE)
ckpt  = os.path.join(MODELS_DIR, "hybrid_fold5_best.pt")
model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
model.eval()
print(f"[OK] Loaded checkpoint: {ckpt}")

# ── Run inference on full val set ─────────────────────────────────────────────
all_preds, all_labels, all_imgs_tensors = [], [], []
with torch.no_grad():
    for imgs, seqs, lbls in val_loader:
        logits = model(imgs.to(DEVICE), seqs.to(DEVICE))
        preds  = logits.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(lbls.numpy())
        all_imgs_tensors.extend(imgs.cpu())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
print(f"[OK] Inference done — {len(all_preds)} samples")
print(f"     Accuracy: {(all_preds == all_labels).mean()*100:.2f}%")

# ── ImageNet un-normalise for display ────────────────────────────────────────
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

def tensor_to_img(t):
    img = t.numpy().transpose(1, 2, 0)
    img = img * STD + MEAN
    return np.clip(img, 0, 1)

# ── Pick 3 samples per class ─────────────────────────────────────────────────
SAMPLES_PER_CLASS = 3
selected = {}   # class_idx -> list of (tensor, gt, pred)

for cls in range(4):
    idxs = np.where(all_labels == cls)[0]
    chosen = idxs[:SAMPLES_PER_CLASS] if len(idxs) >= SAMPLES_PER_CLASS else idxs
    selected[cls] = [(all_imgs_tensors[i], all_labels[i], all_preds[i]) for i in chosen]

# ── Figure 1: Grid — 4 classes × 3 samples ───────────────────────────────────
fig, axes = plt.subplots(4, SAMPLES_PER_CLASS, figsize=(SAMPLES_PER_CLASS * 3.5, 4 * 3.2))
fig.suptitle("Qualitative Results — Proposed CNN+BiLSTM Hybrid\n(Ground Truth  |  Predicted)",
             fontsize=13, fontweight='bold', y=1.01)

for row, cls in enumerate(range(4)):
    for col, (img_t, gt, pred) in enumerate(selected[cls]):
        ax = axes[row][col]
        ax.imshow(tensor_to_img(img_t))
        ax.axis('off')

        gt_name   = CLASS_LABELS[gt]
        pred_name = CLASS_LABELS[pred]
        correct   = (gt == pred)
        border_col = '#27AE60' if correct else '#E74C3C'   # green=correct, red=wrong

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_col)
            spine.set_linewidth(3)

        title_str = f"GT: {gt_name}\nPred: {pred_name}"
        ax.set_title(title_str, fontsize=8.5, color='black',
                     bbox=dict(boxstyle='round,pad=0.3',
                               facecolor=CLASS_COLORS[cls], alpha=0.25))

        if col == 0:
            ax.set_ylabel(CLASS_LABELS[cls].replace('\n', ' '),
                          fontsize=9, fontweight='bold', color=CLASS_COLORS[cls],
                          labelpad=6)

plt.tight_layout()
grid_path = os.path.join(OUT_DIR, "qualitative_grid.png")
plt.savefig(grid_path, dpi=180, bbox_inches='tight')
plt.close()
print(f"[OK] Saved: {grid_path}")

# ── Figure 2: Per-class summary (1 representative image + prediction bar) ────
fig2, axes2 = plt.subplots(1, 4, figsize=(14, 4))
fig2.suptitle("Proposed CNN+BiLSTM — Representative Prediction per Class",
              fontsize=12, fontweight='bold')

for cls in range(4):
    ax = axes2[cls]
    img_t, gt, pred = selected[cls][0]
    ax.imshow(tensor_to_img(img_t))
    ax.axis('off')
    correct = (gt == pred)
    tick = "✓" if correct else "✗"
    col  = '#27AE60' if correct else '#E74C3C'
    ax.set_title(f"{CLASS_LABELS[cls].replace(chr(10),' ')}\n"
                 f"GT: {CLASS_NAMES[gt].split('_')[-1]}\n"
                 f"Pred: {CLASS_NAMES[pred].split('_')[-1]}  {tick}",
                 fontsize=9, color=col, fontweight='bold')
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_edgecolor(col); spine.set_linewidth(3)

plt.tight_layout()
rep_path = os.path.join(OUT_DIR, "qualitative_representative.png")
plt.savefig(rep_path, dpi=180, bbox_inches='tight')
plt.close()
print(f"[OK] Saved: {rep_path}")

# ── Figure 3: Per-class correct/incorrect breakdown ───────────────────────────
fig3, axes3 = plt.subplots(2, 4, figsize=(14, 7))
fig3.suptitle("Proposed CNN+BiLSTM — Correct vs Incorrect Predictions per Class",
              fontsize=12, fontweight='bold')

for cls in range(4):
    cls_idxs   = np.where(all_labels == cls)[0]
    correct_idxs   = [i for i in cls_idxs if all_preds[i] == all_labels[i]]
    incorrect_idxs = [i for i in cls_idxs if all_preds[i] != all_labels[i]]

    # Correct sample
    ax_c = axes3[0][cls]
    if correct_idxs:
        img_t = all_imgs_tensors[correct_idxs[0]]
        gt    = all_labels[correct_idxs[0]]
        pred  = all_preds[correct_idxs[0]]
        ax_c.imshow(tensor_to_img(img_t))
        ax_c.set_title(f"GT: {CLASS_NAMES[gt]}\nPred: {CLASS_NAMES[pred]}", fontsize=7.5, color='#27AE60')
        for sp in ax_c.spines.values(): sp.set_visible(True); sp.set_edgecolor('#27AE60'); sp.set_linewidth(3)
    else:
        ax_c.text(0.5, 0.5, 'All\ncorrect', ha='center', va='center',
                  transform=ax_c.transAxes, fontsize=12, color='#27AE60', fontweight='bold')
    ax_c.axis('off')
    if cls == 0: ax_c.set_ylabel("Correct ✓", fontsize=10, fontweight='bold', color='#27AE60')

    # Incorrect sample
    ax_w = axes3[1][cls]
    if incorrect_idxs:
        img_t = all_imgs_tensors[incorrect_idxs[0]]
        gt    = all_labels[incorrect_idxs[0]]
        pred  = all_preds[incorrect_idxs[0]]
        ax_w.imshow(tensor_to_img(img_t))
        ax_w.set_title(f"GT: {CLASS_NAMES[gt]}\nPred: {CLASS_NAMES[pred]}", fontsize=7.5, color='#E74C3C')
        for sp in ax_w.spines.values(): sp.set_visible(True); sp.set_edgecolor('#E74C3C'); sp.set_linewidth(3)
    else:
        ax_w.text(0.5, 0.5, '0 errors\n(100% acc)', ha='center', va='center',
                  transform=ax_w.transAxes, fontsize=12, color='#27AE60', fontweight='bold')
    ax_w.axis('off')
    if cls == 0: ax_w.set_ylabel("Incorrect ✗", fontsize=10, fontweight='bold', color='#E74C3C')

    axes3[0][cls].set_title(f"{CLASS_LABELS[cls].replace(chr(10),' ')}\n"
                             f"GT: {CLASS_NAMES[all_labels[correct_idxs[0]]].split('_')[-1] if correct_idxs else '-'}\n"
                             f"Pred: {CLASS_NAMES[all_preds[correct_idxs[0]]].split('_')[-1] if correct_idxs else '-'}",
                             fontsize=8, color='#27AE60')

plt.tight_layout()
err_path = os.path.join(OUT_DIR, "qualitative_correct_vs_error.png")
plt.savefig(err_path, dpi=180, bbox_inches='tight')
plt.close()
print(f"[OK] Saved: {err_path}")

print("\n[DONE] All qualitative figures saved to:", OUT_DIR)
print("  qualitative_grid.png           — 4x3 class grid")
print("  qualitative_representative.png — 1 image per class")
print("  qualitative_correct_vs_error.png — correct/incorrect per class")
