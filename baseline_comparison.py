"""
baseline_comparison.py — Baseline Model Comparison
=====================================================
Compares our CNN+BiLSTM Hybrid against 8 baseline models:

Image-only baselines:
  1. VGG16
  2. ResNet50
  3. DenseNet121
  4. EfficientNet-B0
  5. MobileNetV3

Classical ML baselines (on clinical CSV features only):
  6. Random Forest
  7. SVM (RBF kernel)
  8. XGBoost

Our Model:
  9. CNN (EfficientNet-B3) + Bi-LSTM Hybrid  ← proposed

Metrics reported per model:
  Accuracy | Precision | Recall | F1-Score | AUC-ROC | Inference Time
"""

import sys
import os
import time
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Import from main pipeline
from main import (
    OralCancerHybridModel, OralCancerDataset,
    build_image_transforms, preprocess_clinical_csv,
    train_one_epoch, evaluate, CONFIG
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARN] XGBoost not installed. Skipping XGBoost baseline.")


# ─────────────────────────────────────────────
#  SECTION 1: CONFIGURATION
# ─────────────────────────────────────────────

COMPARE_CONFIG = {
    **CONFIG,
    "baseline_epochs": 3,           # fewer epochs for baselines to save time
    "results_dir":     "results",
    "class_names":     ["OSCC", "Leukoplakia_Dysplasia",
                        "Leukoplakia_NoDysplasia", "Normal"],
}

os.makedirs(COMPARE_CONFIG["results_dir"], exist_ok=True)


# ─────────────────────────────────────────────
#  SECTION 2: IMAGE-ONLY BASELINE DATASET
# ─────────────────────────────────────────────

class ImageOnlyDataset(torch.utils.data.Dataset):
    """Minimal image-only dataset for CNN baselines."""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels      = torch.tensor(labels, dtype=torch.long)
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


class ClinicalCNNDataset(torch.utils.data.Dataset):
    """Image + flattened clinical sequence for augmented CNN baselines."""
    def __init__(self, image_paths, sequences, labels, transform=None):
        self.image_paths = image_paths
        self.sequences   = torch.tensor(sequences, dtype=torch.float32)
        self.labels      = torch.tensor(labels,    dtype=torch.long)
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        seq_flat = self.sequences[idx].flatten()   # (seq_len * n_features,)
        return img, seq_flat, self.labels[idx]


class _ClinicalCNNModel(nn.Module):
    """CNN backbone + clinical MLP + fusion head.
    Key difference from our proposed model: no BiLSTM — clinical features
    are flattened and processed by a simple MLP (no temporal modelling)."""
    def __init__(self, backbone, img_feat_dim, clin_dim,
                 num_classes, fusion_dim=256):
        super().__init__()
        self.backbone  = backbone
        self.clin_mlp  = nn.Sequential(
            nn.Linear(clin_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.head = nn.Sequential(
            nn.Linear(img_feat_dim + 128, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, img, clin):
        img_feat  = self.backbone(img)
        clin_feat = self.clin_mlp(clin)
        return self.head(torch.cat([img_feat, clin_feat], dim=1))


# ─────────────────────────────────────────────
#  SECTION 3: CNN BASELINE FACTORY
# ─────────────────────────────────────────────

def build_cnn_baseline(model_name: str, num_classes: int) -> nn.Module:
    """
    Returns a pretrained CNN baseline with replaced classification head.
    Supported: vgg16, resnet50, densenet121, efficientnet_b0, mobilenet_v3
    """
    model_name = model_name.lower()

    if model_name == "vgg16":
        m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        m.classifier[6] = nn.Linear(4096, num_classes)

    elif model_name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    elif model_name == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)

    elif model_name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)

    elif model_name == "mobilenet_v3":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return m


# Output feature dimensions for each backbone (before final classifier)
_CNN_FEAT_DIMS = {
    "vgg16":           4096,
    "resnet50":        2048,
    "densenet121":     1024,
    "efficientnet_b0": 1280,
    "mobilenet_v3":    1280,
}


def build_cnn_extractor(model_name: str):
    """
    Returns (backbone, feature_dim) where the backbone outputs a feature
    vector instead of class logits — used for clinical-fusion baselines.
    """
    model_name = model_name.lower()

    if model_name == "vgg16":
        m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        m.classifier[6] = nn.Identity()

    elif model_name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()

    elif model_name == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        m.classifier = nn.Identity()

    elif model_name == "efficientnet_b0":
        m = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Identity()

    elif model_name == "mobilenet_v3":
        m = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        m.classifier[3] = nn.Identity()

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return m, _CNN_FEAT_DIMS[model_name]


# ─────────────────────────────────────────────
#  SECTION 4: TRAIN & EVALUATE CNN BASELINE
# ─────────────────────────────────────────────

def train_eval_cnn_baseline(
    model_name, image_paths, sequences, labels, cfg, n_folds=3
):
    """
    Trains and evaluates a CNN + clinical-feature baseline using k-fold CV.

    Architecture:
      CNN backbone (pretrained, fine-tuned) --> image feature vector
      Clinical MLP (flattened sequence)     --> clinical feature vector
      Concat --> FusionHead --> 4-class logits

    This is deliberately simpler than our proposed model (no BiLSTM, no
    temporal modelling) — the fusion head is a plain 2-layer MLP.
    Returns averaged metrics dict.
    """
    device       = cfg["device"]
    skf          = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                    random_state=cfg["seed"])
    fold_metrics = []
    clin_dim     = int(sequences.shape[1] * sequences.shape[2])  # 10 * 13 = 130

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(image_paths, labels)):
        tr_imgs  = [image_paths[i] for i in tr_idx]
        vl_imgs  = [image_paths[i] for i in vl_idx]
        tr_seqs, vl_seqs = sequences[tr_idx], sequences[vl_idx]
        tr_lbls,  vl_lbls = labels[tr_idx],   labels[vl_idx]

        train_ds = ClinicalCNNDataset(
            tr_imgs, tr_seqs, tr_lbls,
            transform=build_image_transforms(train=True,  img_size=cfg["img_size"])
        )
        val_ds = ClinicalCNNDataset(
            vl_imgs, vl_seqs, vl_lbls,
            transform=build_image_transforms(train=False, img_size=cfg["img_size"])
        )
        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                                  shuffle=True,  num_workers=cfg.get("num_workers", 0))
        val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                                  shuffle=False, num_workers=cfg.get("num_workers", 0))

        backbone, img_feat_dim = build_cnn_extractor(model_name)
        # Freeze backbone — only the clinical MLP + fusion head are trained.
        # This is ~10x faster on CPU and still uses rich pretrained image features.
        for param in backbone.parameters():
            param.requires_grad = False
        model = _ClinicalCNNModel(
            backbone, img_feat_dim, clin_dim, cfg["num_classes"]
        ).to(device)
        # Optimise only trainable (non-frozen) parameters
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=cfg["lr"],
                                       weight_decay=cfg.get("weight_decay", 1e-5))
        criterion = nn.CrossEntropyLoss()

        for epoch in range(cfg["baseline_epochs"]):
            model.train()
            for imgs, clins, lbls in train_loader:
                imgs, clins, lbls = (imgs.to(device),
                                     clins.to(device),
                                     lbls.to(device))
                optimizer.zero_grad()
                loss = criterion(model(imgs, clins), lbls)
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        start_time = time.time()
        with torch.no_grad():
            for imgs, clins, lbls in val_loader:
                imgs, clins = imgs.to(device), clins.to(device)
                logits = model(imgs, clins)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()
                preds  = logits.argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(lbls.numpy())
                all_probs.extend(probs)
        infer_time = (time.time() - start_time) / len(vl_imgs) * 1000

        metrics = compute_metrics(
            all_labels, all_preds, np.array(all_probs),
            cfg["num_classes"], infer_time
        )
        fold_metrics.append(metrics)
        print(f"    [{model_name}] Fold {fold+1}: "
              f"Acc={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}")

    return average_fold_metrics(fold_metrics)


# ─────────────────────────────────────────────
#  SECTION 5: CLASSICAL ML BASELINES
# ─────────────────────────────────────────────

def train_eval_classical_baseline(
    clf_name, sequences, labels, cfg, n_folds=3
):
    """
    Trains Random Forest / SVM / XGBoost on flattened clinical sequences.
    sequences: (N, seq_len, n_features) → flattened to (N, seq_len * n_features)
    """
    X = sequences.reshape(len(sequences), -1)   # flatten
    y = labels

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                           random_state=cfg["seed"])
    fold_metrics = []

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(X, y)):
        X_tr, X_vl = X[tr_idx], X[vl_idx]
        y_tr, y_vl = y[tr_idx], y[vl_idx]

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_vl   = scaler.transform(X_vl)

        if clf_name == "random_forest":
            clf = RandomForestClassifier(
                n_estimators=200, max_depth=15,
                class_weight="balanced", random_state=cfg["seed"], n_jobs=-1
            )
        elif clf_name == "svm":
            clf = SVC(
                kernel="rbf", C=10, gamma="scale",
                probability=True, class_weight="balanced",
                random_state=cfg["seed"]
            )
        elif clf_name == "xgboost" and XGBOOST_AVAILABLE:
            clf = XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                use_label_encoder=False, eval_metric="mlogloss",
                random_state=cfg["seed"], n_jobs=-1
            )
        else:
            print(f"[SKIP] {clf_name} not available.")
            return None

        start_time = time.time()
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_vl)
        probs = clf.predict_proba(X_vl)
        infer_time = (time.time() - start_time) / len(X_vl) * 1000

        metrics = compute_metrics(y_vl, preds, probs,
                                  cfg["num_classes"], infer_time)
        fold_metrics.append(metrics)
        print(f"    [{clf_name}] Fold {fold+1}: "
              f"Acc={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}")

    return average_fold_metrics(fold_metrics)


# ─────────────────────────────────────────────
#  SECTION 6: EVALUATE PROPOSED HYBRID MODEL
# ─────────────────────────────────────────────

def train_eval_hybrid(image_paths, sequences, labels, cfg, n_folds=3):
    """
    Evaluates the proposed CNN+BiLSTM hybrid model using k-fold CV.
    """
    device = cfg["device"]
    skf    = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=cfg["seed"])
    fold_metrics = []
    n_features = sequences.shape[2]

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(image_paths, labels)):
        tr_imgs  = [image_paths[i] for i in tr_idx]
        vl_imgs  = [image_paths[i] for i in vl_idx]
        tr_seqs, vl_seqs = sequences[tr_idx], sequences[vl_idx]
        tr_lbls, vl_lbls = labels[tr_idx],    labels[vl_idx]

        train_ds = OralCancerDataset(
            tr_imgs, tr_seqs, tr_lbls,
            transform=build_image_transforms(train=True,  img_size=cfg["img_size"])
        )
        val_ds = OralCancerDataset(
            vl_imgs, vl_seqs, vl_lbls,
            transform=build_image_transforms(train=False, img_size=cfg["img_size"])
        )
        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                                  shuffle=True,  num_workers=cfg.get("num_workers", 0))
        val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                                  shuffle=False, num_workers=cfg.get("num_workers", 0))

        model     = OralCancerHybridModel(n_features, cfg).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                       weight_decay=cfg["weight_decay"])

        for epoch in range(cfg["baseline_epochs"]):
            train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Evaluate with timing
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        start_time = time.time()
        with torch.no_grad():
            for imgs, seqs, lbls in val_loader:
                imgs, seqs = imgs.to(device), seqs.to(device)
                logits = model(imgs, seqs)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()
                preds  = logits.argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(lbls.numpy())
                all_probs.extend(probs)
        infer_time = (time.time() - start_time) / len(vl_imgs) * 1000

        metrics = compute_metrics(
            all_labels, all_preds, np.array(all_probs),
            cfg["num_classes"], infer_time
        )
        fold_metrics.append(metrics)
        print(f"    [CNN+BiLSTM] Fold {fold+1}: "
              f"Acc={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}")

    return average_fold_metrics(fold_metrics)


# ─────────────────────────────────────────────
#  SECTION 7: METRICS UTILITIES
# ─────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_proba, num_classes, infer_time_ms):
    """Compute all comparison metrics for one fold."""
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    try:
        auc = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    return {
        "accuracy":   accuracy_score(y_true, y_pred),
        "precision":  precision_score(y_true, y_pred, average="macro",
                                      zero_division=0),
        "recall":     recall_score(y_true, y_pred, average="macro",
                                   zero_division=0),
        "f1":         f1_score(y_true, y_pred, average="macro",
                               zero_division=0),
        "auc_roc":    auc,
        "infer_ms":   infer_time_ms,
        "y_true":     y_true,
        "y_pred":     y_pred,
    }


def average_fold_metrics(fold_metrics: list) -> dict:
    """Average scalar metrics across folds. Keep last fold's predictions."""
    keys = ["accuracy", "precision", "recall", "f1", "auc_roc", "infer_ms"]
    avg  = {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}
    avg["std_accuracy"] = float(np.std([m["accuracy"] for m in fold_metrics]))
    avg["y_true"] = fold_metrics[-1]["y_true"]
    avg["y_pred"] = fold_metrics[-1]["y_pred"]
    return avg


# ─────────────────────────────────────────────
#  SECTION 8: VISUALIZATION
# ─────────────────────────────────────────────

# Color palette: baselines grey, proposed model highlighted
PALETTE = {
    "VGG16":          "#7f8c8d",
    "ResNet50":       "#95a5a6",
    "DenseNet121":    "#bdc3c7",
    "EfficientNet-B0":"#a9cce3",
    "MobileNetV3":    "#abebc6",
    "Random Forest":  "#f0b27a",
    "SVM":            "#f1948a",
    "XGBoost":        "#c39bd3",
    "CNN+BiLSTM\n(Ours)": "#2ecc71",  # highlighted green
}


def plot_accuracy_comparison(results: dict, save_path: str):
    """Bar chart comparing accuracy ± std across all models."""
    names  = list(results.keys())
    accs   = [results[n]["accuracy"] * 100 for n in names]
    stds   = [results[n].get("std_accuracy", 0) * 100 for n in names]
    colors = [PALETTE.get(n, "#7f8c8d") for n in names]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(names, accs, yerr=stds, capsize=5,
                  color=colors, edgecolor="white", linewidth=1.2,
                  error_kw={"ecolor": "#2c3e50", "linewidth": 1.5})

    # Annotate bars
    for bar, acc, std in zip(bars, accs, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.3,
                f"{acc:.2f}%", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#2c3e50")

    ax.set_ylim(60, 102)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Model Accuracy Comparison — Oral Cancer Multi-class Detection",
                 fontsize=13, fontweight="bold", pad=15)
    ax.axhline(y=94, color="#e74c3c", linestyle="--",
               linewidth=1.2, label="94% target line")
    ax.legend(fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_facecolor("#f9f9f9")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {save_path}")


def plot_metrics_radar(results: dict, save_path: str):
    """Radar (spider) chart comparing Accuracy/Precision/Recall/F1/AUC."""
    metrics_keys = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    labels       = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    N = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8),
                           subplot_kw=dict(polar=True))

    colors = list(PALETTE.values())
    model_names = list(results.keys())

    for i, name in enumerate(model_names):
        vals = [results[name].get(k, 0) for k in metrics_keys]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2,
                color=colors[i % len(colors)], label=name)
        ax.fill(angles, vals, alpha=0.05, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["60%", "70%", "80%", "90%", "100%"], fontsize=8)
    ax.set_title("Multi-Metric Radar Comparison", fontsize=13,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1),
              fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {save_path}")


def plot_confusion_matrices(results: dict, class_names: list, save_path: str):
    """Grid of confusion matrices for all models (last fold predictions)."""
    n_models = len(results)
    ncols    = 3
    nrows    = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 5, nrows * 4.5))
    axes = axes.flatten()

    for ax in axes[n_models:]:
        ax.set_visible(False)

    for i, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(res["y_true"], res["y_pred"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[i], linewidths=0.5, cbar=False,
            annot_kws={"size": 9}
        )
        axes[i].set_title(name, fontsize=11, fontweight="bold",
                          color="#2ecc71" if "BiLSTM" in name else "#2c3e50")
        axes[i].set_xlabel("Predicted", fontsize=8)
        axes[i].set_ylabel("Actual",    fontsize=8)
        axes[i].tick_params(axis="x", labelsize=7, rotation=30)
        axes[i].tick_params(axis="y", labelsize=7, rotation=0)

    fig.suptitle("Confusion Matrices — All Models (Last Fold)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {save_path}")


def plot_inference_time(results: dict, save_path: str):
    """Horizontal bar chart of inference time per sample (ms)."""
    names  = list(results.keys())
    times  = [results[n]["infer_ms"] for n in names]
    colors = [PALETTE.get(n, "#7f8c8d") for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, times, color=colors, edgecolor="white")
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{t:.2f} ms", va="center", fontsize=9, color="#2c3e50")
    ax.set_xlabel("Inference Time per Sample (ms)", fontsize=11)
    ax.set_title("Inference Speed Comparison", fontsize=13,
                 fontweight="bold", pad=12)
    ax.set_facecolor("#f9f9f9")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {save_path}")


def plot_f1_per_class(results: dict, class_names: list, save_path: str):
    """
    Grouped bar chart showing per-class F1 for each model.
    Highlights performance on minority classes (leukoplakia).
    """
    model_names = list(results.keys())
    n_classes   = len(class_names)
    x           = np.arange(n_classes)
    width        = 0.08
    offsets      = np.linspace(-(len(model_names)-1)*width/2,
                                (len(model_names)-1)*width/2,
                                len(model_names))
    fig, ax = plt.subplots(figsize=(13, 6))
    colors  = list(PALETTE.values())

    for i, name in enumerate(model_names):
        report = classification_report(
            results[name]["y_true"],
            results[name]["y_pred"],
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        f1s = [report[c]["f1-score"] for c in class_names]
        ax.bar(x + offsets[i], f1s, width,
               label=name, color=colors[i % len(colors)],
               edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_ylabel("F1-Score", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title("Per-Class F1-Score Comparison", fontsize=13,
                 fontweight="bold", pad=12)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_facecolor("#f9f9f9")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {save_path}")


# ─────────────────────────────────────────────
#  SECTION 9: SUMMARY TABLE
# ─────────────────────────────────────────────

def generate_summary_table(results: dict, save_path: str):
    """
    Saves a formatted CSV + styled HTML summary table of all model metrics.
    Highlights the best value in each column.
    """
    rows = []
    for name, res in results.items():
        rows.append({
            "Model":         name,
            "Accuracy (%)":  round(res["accuracy"] * 100, 2),
            "Precision (%)": round(res["precision"] * 100, 2),
            "Recall (%)":    round(res["recall"] * 100, 2),
            "F1-Score (%)":  round(res["f1"] * 100, 2),
            "AUC-ROC":       round(res["auc_roc"], 4),
            "Infer (ms)":    round(res["infer_ms"], 3),
            "Std Acc (%)":   round(res.get("std_accuracy", 0) * 100, 2),
        })
    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = save_path.replace(".html", ".csv")
    df.to_csv(csv_path, index=False)
    print(f"[OK] Summary CSV saved: {csv_path}")

    # Styled HTML
    def highlight_best(col):
        if col.name in ["Accuracy (%)", "Precision (%)",
                        "Recall (%)", "F1-Score (%)", "AUC-ROC"]:
            is_best = col == col.max()
        elif col.name in ["Infer (ms)", "Std Acc (%)"]:
            is_best = col == col.min()
        else:
            return [""] * len(col)
        return ["background-color: #d5f5e3; font-weight: bold"
                if v else "" for v in is_best]

    styled = (df.style
              .apply(highlight_best)
              .set_caption("Oral Cancer Detection — Model Comparison Table")
              .set_table_styles([{
                  "selector": "caption",
                  "props": [("font-size", "14px"), ("font-weight", "bold"),
                             ("color", "#2c3e50"), ("padding-bottom", "10px")]
              }]))
    styled.to_html(save_path)
    print(f"[OK] Summary HTML saved: {save_path}")
    print(f"\n{'='*65}")
    print(df.to_string(index=False))
    print(f"{'='*65}\n")
    return df


# ─────────────────────────────────────────────
#  SECTION 10: MAIN COMPARISON RUNNER
# ─────────────────────────────────────────────

def run_full_comparison(image_paths, sequences, labels, cfg):
    """
    Runs all 9 models and collects metrics.
    Returns dict: model_name → metrics dict
    """
    results = {}
    n_folds = 3   # use 3 folds for baselines (speed); 5 for final paper results

    # ── CNN-only baselines ──────────────────────────
    cnn_baselines = {
        "VGG16":           "vgg16",
        "ResNet50":        "resnet50",
        "DenseNet121":     "densenet121",
        "EfficientNet-B0": "efficientnet_b0",
        "MobileNetV3":     "mobilenet_v3",
    }
    for display_name, model_key in cnn_baselines.items():
        print(f"\n[Running] {display_name}...")
        results[display_name] = train_eval_cnn_baseline(
            model_key, image_paths, sequences, labels, cfg, n_folds=n_folds
        )

    # ── Classical ML baselines ──────────────────────
    classical_baselines = {
        "Random Forest": "random_forest",
        "SVM":           "svm",
    }
    if XGBOOST_AVAILABLE:
        classical_baselines["XGBoost"] = "xgboost"

    for display_name, clf_key in classical_baselines.items():
        print(f"\n[Running] {display_name}...")
        res = train_eval_classical_baseline(
            clf_key, sequences, labels, cfg, n_folds=n_folds
        )
        if res:
            results[display_name] = res

    # ── Proposed hybrid model ────────────────────────
    print(f"\n[Running] CNN+BiLSTM (Proposed)...")
    results["CNN+BiLSTM\n(Ours)"] = train_eval_hybrid(
        image_paths, sequences, labels, cfg, n_folds=n_folds
    )

    return results


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "="*65)
    print("  Oral Cancer Detection — Baseline Comparison Suite")
    print("="*65)

    # --- Load data (same as main.py) ---
    print("\n[1] Loading clinical sequences...")
    sequences, labels, _ = preprocess_clinical_csv(
        COMPARE_CONFIG["clinical_csv"],
        seq_len=COMPARE_CONFIG["lstm_seq_len"]
    )

    print("\n[2] Building image path list...")
    from glob import glob
    image_paths = (
        glob(os.path.join(COMPARE_CONFIG["mendeley_img_dir"],
                          "**/*.jpg"), recursive=True) +
        glob(os.path.join(COMPARE_CONFIG["ndbufes_img_dir"],
                          "**/*.png"), recursive=True)
    )
    print(f"    Total images: {len(image_paths)}")

    # --- Run all models ---
    print("\n[3] Running comparison (this may take a while)...")
    results = run_full_comparison(
        image_paths, sequences, labels, COMPARE_CONFIG
    )

    # --- Generate all plots ---
    print("\n[4] Generating visualizations...")
    rd = COMPARE_CONFIG["results_dir"]

    plot_accuracy_comparison(
        results,
        save_path=f"{rd}/accuracy_comparison.png"
    )
    plot_metrics_radar(
        results,
        save_path=f"{rd}/radar_comparison.png"
    )
    plot_confusion_matrices(
        results, COMPARE_CONFIG["class_names"],
        save_path=f"{rd}/confusion_matrices.png"
    )
    plot_inference_time(
        results,
        save_path=f"{rd}/inference_time.png"
    )
    plot_f1_per_class(
        results, COMPARE_CONFIG["class_names"],
        save_path=f"{rd}/f1_per_class.png"
    )

    # --- Summary table ---
    print("\n[5] Generating summary table...")
    df_summary = generate_summary_table(
        results,
        save_path=f"{rd}/model_comparison_table.html"
    )

    print(f"\n[OK] All results saved to ./{rd}/")
    print("    Files generated:")
    print("      accuracy_comparison.png")
    print("      radar_comparison.png")
    print("      confusion_matrices.png")
    print("      inference_time.png")
    print("      f1_per_class.png")
    print("      model_comparison_table.html")
    print("      model_comparison_table.csv")
