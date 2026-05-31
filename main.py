"""
Oral Cancer Detection: CNN + Bi-LSTM Hybrid Fusion Pipeline
============================================================
Datasets used:
  - Mendeley OSCC Histopathology (image stream)
  - NDB-UFES (image + CSV stream)
  - OCDC segmentation masks (ROI preprocessing)
  - TCGA / Kaggle clinical CSVs (LSTM stream)

Classes: OSCC | Leukoplakia_Dysplasia | Leukoplakia_NoDysplasia | Normal
Target accuracy: 94-96%
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  SECTION 1: CONFIGURATION
# ─────────────────────────────────────────────

CONFIG = {
    # Paths — update these to your local dataset locations
    "mendeley_img_dir":  "data/mendeley/images",
    "ndbufes_img_dir":   "data/ndbufes/images",
    "clinical_csv":      "data/clinical/merged_clinical.csv",
    "mask_dir":          "data/ocdc/masks",          # optional ROI masks

    # Model hyperparameters
    "img_size":          224,
    "cnn_feature_dim":   512,
    "lstm_hidden":       128,
    "lstm_layers":       2,
    "lstm_seq_len":      10,      # number of clinical time-steps per patient
    "num_classes":       4,
    "dropout":           0.4,
    "fusion_dim":        256,

    # Training
    "batch_size":        32,
    "epochs":            50,
    "lr":                1e-4,
    "weight_decay":      1e-5,
    "n_folds":           5,
    "seed":              42,

    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Class names (must match label encoding order)
    "class_names": ["OSCC", "Leukoplakia_Dysplasia", "Leukoplakia_NoDysplasia", "Normal"],
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])


# ─────────────────────────────────────────────
#  SECTION 2: DATA PREPROCESSING
# ─────────────────────────────────────────────

class StainNormalizer:
    """
    Macenko stain normalization for H&E histopathology images.
    Reduces domain shift across Mendeley / NDB-UFES / OCDC sources.
    Install: pip install staintools
    """
    def __init__(self):
        try:
            import staintools
            self.normalizer = staintools.StainNormalizer(method="macenko")
            self._fitted = False
        except ImportError:
            print("[WARN] staintools not installed. Skipping stain normalization.")
            self.normalizer = None

    def fit(self, reference_image_path: str):
        if self.normalizer:
            ref = staintools.read_image(reference_image_path)
            self.normalizer.fit(ref)
            self._fitted = True

    def transform(self, image: np.ndarray) -> np.ndarray:
        if self.normalizer and self._fitted:
            try:
                return self.normalizer.transform(image)
            except Exception:
                return image
        return image


def apply_roi_mask(image: np.ndarray, mask_path: str) -> np.ndarray:
    """
    Crop image to tumor ROI using OCDC segmentation masks.
    Falls back to full image if mask not available.
    """
    if mask_path and os.path.exists(mask_path):
        mask = np.array(Image.open(mask_path).convert("L"))
        coords = np.argwhere(mask > 0)
        if len(coords):
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            return image[y0:y1, x0:x1]
    return image


def build_image_transforms(train: bool = True, img_size: int = 224):
    """
    Image augmentation pipeline.
    Training: aggressive augmentation to handle class imbalance.
    Validation: only resize + normalize.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    if train:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])


def preprocess_clinical_csv(csv_path: str, seq_len: int = 10):
    """
    Load and preprocess clinical/tabular data for LSTM input.

    Expected CSV columns (adapt to your merged dataset):
      patient_id, age, gender, tobacco_use, alcohol_use, betel_nut,
      hpv_status, ulcer_size, pain_score, visit_month, label

    Returns:
      sequences : np.ndarray of shape (N, seq_len, n_features)
      labels    : np.ndarray of shape (N,)
      patient_ids: list
    """
    df = pd.read_csv(csv_path)

    # --- Encode categoricals ---
    cat_cols = ["gender", "tobacco_use", "alcohol_use",
                "betel_nut", "hpv_status"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # --- Encode labels ---
    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"])

    feature_cols = [c for c in df.columns
                    if c not in ["patient_id", "label", "label_enc", "visit_month"]]

    # --- Scale features ---
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # --- Build sequences per patient (sorted by visit_month) ---
    sequences, labels, pids = [], [], []
    for pid, group in df.groupby("patient_id"):
        group = group.sort_values("visit_month")
        feats = group[feature_cols].values  # (T, F)
        label = group["label_enc"].iloc[-1]

        # Pad or truncate to seq_len
        T, F = feats.shape
        if T >= seq_len:
            seq = feats[-seq_len:]
        else:
            pad = np.zeros((seq_len - T, F))
            seq = np.vstack([pad, feats])

        sequences.append(seq)
        labels.append(label)
        pids.append(pid)

    return np.array(sequences, dtype=np.float32), np.array(labels), pids


# ─────────────────────────────────────────────
#  SECTION 3: DATASET CLASS
# ─────────────────────────────────────────────

class OralCancerDataset(Dataset):
    """
    Multi-modal dataset: returns (image_tensor, clinical_sequence, label).

    image_paths   : list of paths to histopathology images
    sequences     : np.ndarray (N, seq_len, n_features) — clinical LSTM input
    labels        : np.ndarray (N,) — integer class labels
    mask_paths    : optional list of ROI mask paths (same length as image_paths)
    transform     : torchvision transform pipeline
    stain_norm    : StainNormalizer instance (optional)
    """
    def __init__(self, image_paths, sequences, labels,
                 mask_paths=None, transform=None, stain_norm=None):
        self.image_paths = image_paths
        self.sequences   = torch.tensor(sequences, dtype=torch.float32)
        self.labels      = torch.tensor(labels, dtype=torch.long)
        self.mask_paths  = mask_paths or [None] * len(image_paths)
        self.transform   = transform
        self.stain_norm  = stain_norm

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # --- Load image ---
        img = np.array(Image.open(self.image_paths[idx]).convert("RGB"))

        # --- Optional: ROI crop ---
        img = apply_roi_mask(img, self.mask_paths[idx])

        # --- Optional: stain normalization ---
        if self.stain_norm:
            img = self.stain_norm.transform(img)

        img = Image.fromarray(img.astype(np.uint8))

        if self.transform:
            img = self.transform(img)

        seq   = self.sequences[idx]      # (seq_len, n_features)
        label = self.labels[idx]

        return img, seq, label


# ─────────────────────────────────────────────
#  SECTION 4: MODEL ARCHITECTURE
# ─────────────────────────────────────────────

class CNNEncoder(nn.Module):
    """
    EfficientNet-B3 backbone pretrained on ImageNet.
    Outputs a fixed-size feature vector for each image.
    Swap backbone here if needed (DenseNet121, ResNet50, etc.)
    """
    def __init__(self, feature_dim: int = 512, freeze_base: bool = False):
        super().__init__()
        from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
        base = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)

        if freeze_base:
            for param in base.parameters():
                param.requires_grad = False

        # Replace classifier head
        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, feature_dim),
            nn.ReLU(),
        )
        self.backbone = base

    def forward(self, x):
        return self.backbone(x)  # (B, feature_dim)


class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM for encoding sequential clinical patient data.
    Takes (B, seq_len, n_features) → returns (B, lstm_hidden * 2)
    """
    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x):
        out, (h, _) = self.lstm(x)
        # Concatenate final forward and backward hidden states
        h_fwd = h[-2]   # last layer, forward
        h_bwd = h[-1]   # last layer, backward
        fused = torch.cat([h_fwd, h_bwd], dim=1)  # (B, hidden*2)
        return self.layer_norm(fused)


class FusionClassifier(nn.Module):
    """
    Late fusion: concatenate CNN + Bi-LSTM feature vectors,
    then pass through MLP head for multi-class prediction.

    Architecture:
      [CNN(512)] + [BiLSTM(256)] → concat(768) → Dense(256) → Dense(4)
    """
    def __init__(self, cnn_dim: int, lstm_dim: int,
                 fusion_dim: int, num_classes: int, dropout: float = 0.4):
        super().__init__()
        input_dim = cnn_dim + lstm_dim
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(fusion_dim // 2, num_classes),
        )

    def forward(self, cnn_feat, lstm_feat):
        x = torch.cat([cnn_feat, lstm_feat], dim=1)
        return self.fusion(x)


class OralCancerHybridModel(nn.Module):
    """
    Full CNN + Bi-LSTM hybrid model for oral cancer multi-class detection.

    Forward pass:
      image  → CNNEncoder  → cnn_features (B, 512)
      seq    → BiLSTMEncoder → lstm_features (B, 256)
      concat → FusionClassifier → logits (B, 4)
    """
    def __init__(self, lstm_input_size: int, cfg: dict):
        super().__init__()
        self.cnn  = CNNEncoder(feature_dim=cfg["cnn_feature_dim"])
        self.lstm = BiLSTMEncoder(
            input_size=lstm_input_size,
            hidden_size=cfg["lstm_hidden"],
            num_layers=cfg["lstm_layers"],
        )
        lstm_out_dim = cfg["lstm_hidden"] * 2  # bidirectional
        self.classifier = FusionClassifier(
            cnn_dim=cfg["cnn_feature_dim"],
            lstm_dim=lstm_out_dim,
            fusion_dim=cfg["fusion_dim"],
            num_classes=cfg["num_classes"],
            dropout=cfg["dropout"],
        )

    def forward(self, image, sequence):
        cnn_feat  = self.cnn(image)      # (B, 512)
        lstm_feat = self.lstm(sequence)  # (B, 256)
        logits    = self.classifier(cnn_feat, lstm_feat)
        return logits


# ─────────────────────────────────────────────
#  SECTION 5: TRAINING UTILITIES
# ─────────────────────────────────────────────

def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Inverse-frequency class weights to handle leukoplakia class imbalance.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, seqs, labels in loader:
        imgs, seqs, labels = imgs.to(device), seqs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs, seqs)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, seqs, labels in loader:
        imgs, seqs, labels = imgs.to(device), seqs.to(device), labels.to(device)
        logits = model(imgs, seqs)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


# ─────────────────────────────────────────────
#  SECTION 6: 5-FOLD CROSS VALIDATION TRAINING
# ─────────────────────────────────────────────

def run_training(image_paths, sequences, labels, cfg):
    """
    5-fold stratified cross-validation training loop.
    Saves the best model checkpoint per fold.
    """
    device = cfg["device"]
    skf    = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True,
                              random_state=cfg["seed"])
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels)):
        print(f"\n{'='*55}")
        print(f"  FOLD {fold+1} / {cfg['n_folds']}")
        print(f"{'='*55}")

        # --- Split data ---
        tr_imgs  = [image_paths[i] for i in train_idx]
        vl_imgs  = [image_paths[i] for i in val_idx]
        tr_seqs, vl_seqs = sequences[train_idx], sequences[val_idx]
        tr_lbls, vl_lbls = labels[train_idx],    labels[val_idx]

        # --- Datasets & loaders ---
        train_ds = OralCancerDataset(
            tr_imgs, tr_seqs, tr_lbls,
            transform=build_image_transforms(train=True,  img_size=cfg["img_size"])
        )
        val_ds = OralCancerDataset(
            vl_imgs, vl_seqs, vl_lbls,
            transform=build_image_transforms(train=False, img_size=cfg["img_size"])
        )
        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                                  shuffle=True,  num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                                  shuffle=False, num_workers=2, pin_memory=True)

        # --- Model, optimizer, scheduler ---
        n_features = sequences.shape[2]
        model = OralCancerHybridModel(lstm_input_size=n_features, cfg=cfg).to(device)

        class_weights = compute_class_weights(tr_lbls, cfg["num_classes"]).to(device)
        criterion     = nn.CrossEntropyLoss(weight=class_weights)
        optimizer     = torch.optim.AdamW(model.parameters(),
                                          lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler     = torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, T_max=cfg["epochs"], eta_min=1e-6)

        best_val_acc, best_epoch = 0.0, 0
        checkpoint_path = f"checkpoints/fold{fold+1}_best.pt"
        os.makedirs("checkpoints", exist_ok=True)

        for epoch in range(1, cfg["epochs"] + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer,
                                               criterion, device)
            vl_loss, vl_acc, preds, gts = evaluate(model, val_loader,
                                                     criterion, device)
            scheduler.step()

            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                best_epoch   = epoch
                torch.save(model.state_dict(), checkpoint_path)

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{cfg['epochs']} | "
                      f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f} | "
                      f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f}")

        # --- Final evaluation on best model ---
        model.load_state_dict(torch.load(checkpoint_path))
        _, final_acc, final_preds, final_gts = evaluate(model, val_loader,
                                                          criterion, device)
        print(f"\n  Best Val Acc (Fold {fold+1}): {best_val_acc:.4f} @ Epoch {best_epoch}")
        print(f"\n  Classification Report:\n")
        print(classification_report(final_gts, final_preds,
                                    target_names=cfg["class_names"]))
        fold_results.append(best_val_acc)

    print(f"\n{'='*55}")
    print(f"  5-Fold CV Results: {[f'{a:.4f}' for a in fold_results]}")
    print(f"  Mean Accuracy : {np.mean(fold_results):.4f}")
    print(f"  Std Deviation : {np.std(fold_results):.4f}")
    print(f"{'='*55}")
    return fold_results


# ─────────────────────────────────────────────
#  SECTION 7: INFERENCE PIPELINE
# ─────────────────────────────────────────────

def predict_single(model, image_path: str, clinical_sequence: np.ndarray,
                   cfg: dict, class_names: list):
    """
    Predict class for a single patient.

    Args:
        image_path       : path to histopathology image
        clinical_sequence: np.ndarray of shape (seq_len, n_features)
    Returns:
        predicted class name and probabilities
    """
    device = cfg["device"]
    model.eval().to(device)

    transform = build_image_transforms(train=False, img_size=cfg["img_size"])
    img  = Image.open(image_path).convert("RGB")
    img  = transform(img).unsqueeze(0).to(device)      # (1, 3, H, W)
    seq  = torch.tensor(clinical_sequence,
                        dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, F)

    with torch.no_grad():
        logits = model(img, seq)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred   = np.argmax(probs)

    print(f"\n  Predicted Class : {class_names[pred]}")
    print(f"  Confidence      : {probs[pred]*100:.2f}%")
    for i, name in enumerate(class_names):
        print(f"    {name:30s}: {probs[i]*100:.2f}%")
    return class_names[pred], probs


# ─────────────────────────────────────────────
#  SECTION 8: ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("\n[1] Loading and preprocessing clinical CSV...")
    sequences, labels, patient_ids = preprocess_clinical_csv(
        CONFIG["clinical_csv"], seq_len=CONFIG["lstm_seq_len"]
    )
    print(f"    Sequences shape : {sequences.shape}")
    print(f"    Labels shape    : {labels.shape}")
    print(f"    Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    print("\n[2] Building image path list...")
    # Combine Mendeley + NDB-UFES image paths
    # Adapt these glob patterns to match your directory structure
    from glob import glob
    image_paths = (
        glob(os.path.join(CONFIG["mendeley_img_dir"], "**/*.jpg"), recursive=True) +
        glob(os.path.join(CONFIG["ndbufes_img_dir"],  "**/*.png"), recursive=True)
    )
    print(f"    Total images found: {len(image_paths)}")

    # NOTE: In practice, align image_paths with patient_ids from the CSV.
    # The sequences and image_paths must correspond to the same patients (same N).
    # Use a merge on patient_id between your CSV metadata and image filenames.

    print("\n[3] Starting 5-Fold Cross-Validation...")
    fold_results = run_training(image_paths, sequences, labels, CONFIG)

    print("\n[4] Training complete. Checkpoints saved in ./checkpoints/")
    print("    Use predict_single() for inference on new patients.")
