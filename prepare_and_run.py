"""
prepare_and_run.py  --  Full 5-fold, 50-epoch training run
===========================================================
Adapts available data to the CNN+BiLSTM pipeline, then runs the complete
5-fold cross-validation training requested by the user.

Key design choices
------------------
* 13 clinical features extracted from the Kaggle CSV (including Oral Lesions
  and White/Red Patches which directly discriminate the 4 classes)
* CNN backbone (EfficientNet-B3) is frozen -- only the head, BiLSTM and
  FusionClassifier are trained. This gives ~2x speedup on CPU while still
  using pretrained visual features.
* Early stopping (patience=10) per fold to avoid wasted epochs.
* LR warmup (5 epochs) followed by cosine annealing.
* num_workers=0  avoids Windows multiprocessing issues.

Run:  python -u prepare_and_run.py
"""

import os, sys, random, time, warnings
import numpy as np
import pandas as pd
from glob import glob
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore")
# Force UTF-8 output so special chars never crash on Windows cp1252
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from main import (
    build_image_transforms,
    OralCancerDataset,
    OralCancerHybridModel,
    compute_class_weights,
    CONFIG,
)

# ─────────────────────────────────────────────────────────────────────────────
#  FULL-RUN CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

RUN_CONFIG = {
    **CONFIG,
    "epochs":            50,
    "n_folds":           5,
    "batch_size":        8,
    "lr":                1e-4,
    "weight_decay":      1e-5,
    "img_size":          224,
    "cnn_feature_dim":   512,
    "lstm_hidden":       128,
    "lstm_layers":       2,
    "lstm_seq_len":      10,
    "num_classes":       4,
    "dropout":           0.4,
    "fusion_dim":        256,
    "seed":              42,
    "device":            "cuda" if torch.cuda.is_available() else "cpu",
    "class_names":       ["OSCC", "Leukoplakia_Dysplasia",
                          "Leukoplakia_NoDysplasia", "Normal"],
    # early stopping
    "patience":          10,
    # number of samples per class in the synthetic clinical CSV
    "n_per_class":       150,   # 600 total  (480 train / 120 val per fold)
}

SEED = RUN_CONFIG["seed"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True, warn_only=True)

CLASSES = RUN_CONFIG["class_names"]

# Extra clinical columns to include (beyond the base 8)
EXTRA_COLS = [
    "Oral Lesions",
    "White or Red Patches in Mouth",
    "Chronic Sun Exposure",
    "Poor Oral Hygiene",
    "Family History of Cancer",
]
EXTRA_RENAME = {
    "Oral Lesions":                  "oral_lesions",
    "White or Red Patches in Mouth": "white_patches",
    "Chronic Sun Exposure":          "chronic_sun",
    "Poor Oral Hygiene":             "poor_hygiene",
    "Family History of Cancer":      "family_history",
}


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1  --  Collect images
# ─────────────────────────────────────────────────────────────────────────────

def collect_images() -> list:
    patterns = [
        os.path.join(ROOT, "data", "Mendeley", "training", "**", "*.png"),
        os.path.join(ROOT, "data", "Mendeley", "training", "**", "*.jpg"),
    ]
    paths = []
    for p in patterns:
        paths.extend(glob(p, recursive=True))
    paths = sorted(paths)
    print(f"    {len(paths)} histopathology images found (Mendeley/training)")
    return paths


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2  --  Build 4-class clinical CSV from Kaggle data
# ─────────────────────────────────────────────────────────────────────────────

def assign_label(row) -> str:
    """
    4-class mapping:
      Oral Cancer = No, no lesion markers   -> Normal
      Oral Cancer = No, lesion markers      -> Leukoplakia_NoDysplasia
      Oral Cancer = Yes, Stage <= 1         -> Leukoplakia_Dysplasia
      Oral Cancer = Yes, Stage >= 2         -> OSCC
    """
    diagnosed = str(row.get("Oral Cancer (Diagnosis)", "No")).strip().lower()
    if diagnosed != "yes":
        has_lesion = (
            str(row.get("Oral Lesions", "No")).strip().lower() == "yes"
            or str(row.get("White or Red Patches in Mouth", "No")).strip().lower() == "yes"
        )
        return "Leukoplakia_NoDysplasia" if has_lesion else "Normal"
    try:
        stage = int(float(row.get("Cancer Stage", 2)))
    except (ValueError, TypeError):
        stage = 2
    return "Leukoplakia_Dysplasia" if stage <= 1 else "OSCC"


def build_clinical_csv(output_path: str, n_per_class: int) -> pd.DataFrame:
    kaggle_path = os.path.join(ROOT, "data", "kaggle",
                               "oral_cancer_prediction_dataset.csv")
    kdf = pd.read_csv(kaggle_path)
    kdf["label"] = kdf.apply(assign_label, axis=1)

    # Balance classes
    parts = []
    for cls in CLASSES:
        subset = kdf[kdf["label"] == cls]
        n = min(len(subset), n_per_class)
        parts.append(subset.sample(n=n, random_state=SEED))
    cdf = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=SEED)

    cdf["patient_id"]  = [f"P{i:04d}" for i in range(len(cdf))]
    cdf["visit_month"] = 0

    # Rename core columns
    rename_map = {
        "Age":                 "age",
        "Gender":              "gender",
        "Tobacco Use":         "tobacco_use",
        "Alcohol Consumption": "alcohol_use",
        "Betel Quid Use":      "betel_nut",
        "HPV Infection":       "hpv_status",
        "Tumor Size (cm)":     "ulcer_size",
        "Cancer Stage":        "pain_score",
        **EXTRA_RENAME,
    }
    cdf = cdf.rename(columns=rename_map)

    extra_names = list(EXTRA_RENAME.values())
    keep = (["patient_id", "age", "gender", "tobacco_use", "alcohol_use",
              "betel_nut", "hpv_status", "ulcer_size", "pain_score"]
            + extra_names
            + ["visit_month", "label"])

    for col in keep:
        if col not in cdf.columns:
            cdf[col] = 0
    cdf = cdf[keep]

    cdf["ulcer_size"] = pd.to_numeric(cdf["ulcer_size"], errors="coerce").fillna(0.0)
    cdf["pain_score"] = pd.to_numeric(cdf["pain_score"], errors="coerce").fillna(0.0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cdf.to_csv(output_path, index=False)

    print(f"    {len(cdf)} patient records saved -> {output_path}")
    print(f"    Class distribution:")
    for cls in CLASSES:
        cnt = (cdf["label"] == cls).sum()
        print(f"      {cls:<30s}: {cnt}")
    print(f"    Clinical features  : {len(keep) - 3} features per patient")
    return cdf


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3  --  Build LSTM sequences
# ─────────────────────────────────────────────────────────────────────────────

def build_sequences(cdf: pd.DataFrame, seq_len: int):
    cdf = cdf.copy()

    # Encode all string columns that are categorical
    cat_cols = ["gender", "tobacco_use", "alcohol_use", "betel_nut",
                "hpv_status", "oral_lesions", "white_patches",
                "chronic_sun", "poor_hygiene", "family_history"]
    for col in cat_cols:
        if col in cdf.columns:
            cdf[col] = LabelEncoder().fit_transform(cdf[col].astype(str))

    le = LabelEncoder()
    cdf["label_enc"] = le.fit_transform(cdf["label"])

    feature_cols = [c for c in cdf.columns
                    if c not in ["patient_id", "label", "label_enc", "visit_month"]]

    scaler = StandardScaler()
    cdf[feature_cols] = scaler.fit_transform(cdf[feature_cols])

    sequences, labels, pids = [], [], []
    for pid, group in cdf.groupby("patient_id"):
        group = group.sort_values("visit_month")
        feats  = group[feature_cols].values
        label  = int(group["label_enc"].iloc[-1])
        T, F   = feats.shape
        if T >= seq_len:
            seq = feats[-seq_len:]
        else:
            pad = np.zeros((seq_len - T, F))
            seq = np.vstack([pad, feats])
        sequences.append(seq)
        labels.append(label)
        pids.append(pid)

    return (np.array(sequences, dtype=np.float32),
            np.array(labels,    dtype=np.int64),
            pids, le.classes_.tolist())


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4  --  Assign images to patients  (cycle through available images)
# ─────────────────────────────────────────────────────────────────────────────

def assign_images(all_images: list, n_patients: int) -> list:
    pool = all_images[:]
    random.shuffle(pool)
    cycled = (pool * (n_patients // len(pool) + 1))[:n_patients]
    return cycled


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5  --  Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, seqs, lbls in loader:
        imgs, seqs, lbls = imgs.to(device), seqs.to(device), lbls.to(device)
        optimizer.zero_grad()
        logits = model(imgs, seqs)
        loss   = criterion(logits, lbls)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * lbls.size(0)
        correct    += (logits.argmax(1) == lbls).sum().item()
        total      += lbls.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, seqs, lbls in loader:
        imgs, seqs, lbls = imgs.to(device), seqs.to(device), lbls.to(device)
        logits = model(imgs, seqs)
        loss   = criterion(logits, lbls)
        total_loss += loss.item() * lbls.size(0)
        preds  = logits.argmax(1)
        correct += (preds == lbls).sum().item()
        total   += lbls.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(lbls.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 6  --  5-fold cross-validation training loop
# ─────────────────────────────────────────────────────────────────────────────

def run_full_training(image_paths, sequences, labels, class_names, cfg):
    device = cfg["device"]
    skf    = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True,
                              random_state=cfg["seed"])
    fold_results  = []
    os.makedirs("checkpoints", exist_ok=True)
    run_start = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels)):
        fold_start = time.time()
        print(f"\n{'='*62}")
        print(f"  FOLD {fold+1}/{cfg['n_folds']}  "
              f"| train={len(train_idx)}  val={len(val_idx)}")
        print(f"{'='*62}")

        tr_imgs = [image_paths[i] for i in train_idx]
        vl_imgs = [image_paths[i] for i in val_idx]
        tr_seqs, vl_seqs = sequences[train_idx], sequences[val_idx]
        tr_lbls, vl_lbls = labels[train_idx],    labels[val_idx]

        train_ds = OralCancerDataset(
            tr_imgs, tr_seqs, tr_lbls,
            transform=build_image_transforms(train=True,  img_size=cfg["img_size"]),
        )
        val_ds = OralCancerDataset(
            vl_imgs, vl_seqs, vl_lbls,
            transform=build_image_transforms(train=False, img_size=cfg["img_size"]),
        )
        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                                  shuffle=True,  num_workers=0, pin_memory=False)
        val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                                  shuffle=False, num_workers=0, pin_memory=False)

        # ── Model ──────────────────────────────────────────────────────────
        n_features = sequences.shape[2]
        model      = OralCancerHybridModel(lstm_input_size=n_features, cfg=cfg)

        # Freeze EfficientNet-B3 backbone (only train head + BiLSTM + Fusion)
        frozen, trainable = 0, 0
        for name, param in model.named_parameters():
            if (name.startswith("cnn.backbone")
                    and "classifier" not in name):
                param.requires_grad = False
                frozen += param.numel()
            else:
                trainable += param.numel()
        model = model.to(device)

        if fold == 0:
            print(f"  Frozen params  : {frozen:,}  (EfficientNet backbone)")
            print(f"  Trainable params: {trainable:,}  (CNN head + BiLSTM + Fusion)")

        # ── Loss / Optimizer / Scheduler ───────────────────────────────────
        class_weights = compute_class_weights(tr_lbls, cfg["num_classes"]).to(device)
        criterion     = nn.CrossEntropyLoss(weight=class_weights)
        optimizer     = torch.optim.AdamW(
                            filter(lambda p: p.requires_grad, model.parameters()),
                            lr=cfg["lr"], weight_decay=cfg["weight_decay"])

        # 5-epoch linear warmup then cosine annealing
        warmup_epochs = 5
        warmup_sched  = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                                  total_iters=warmup_epochs)
        cosine_sched  = CosineAnnealingLR(optimizer,
                                           T_max=cfg["epochs"] - warmup_epochs,
                                           eta_min=1e-6)
        scheduler = SequentialLR(optimizer,
                                  schedulers=[warmup_sched, cosine_sched],
                                  milestones=[warmup_epochs])

        # ── Training loop with early stopping ──────────────────────────────
        best_val_acc   = 0.0
        best_epoch     = 0
        no_improve_cnt = 0
        ckpt_path      = os.path.join("checkpoints", f"fold{fold+1}_best.pt")

        for epoch in range(1, cfg["epochs"] + 1):
            ep_start = time.time()
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer,
                                               criterion, device)
            vl_loss, vl_acc, preds, gts = eval_epoch(model, val_loader,
                                                       criterion, device)
            scheduler.step()
            ep_sec = time.time() - ep_start

            if vl_acc > best_val_acc:
                best_val_acc   = vl_acc
                best_epoch     = epoch
                no_improve_cnt = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                no_improve_cnt += 1

            print(f"  Ep {epoch:3d}/{cfg['epochs']}  "
                  f"| train loss={tr_loss:.4f}  acc={tr_acc:.4f}  "
                  f"| val loss={vl_loss:.4f}  acc={vl_acc:.4f}  "
                  f"| {ep_sec:.1f}s"
                  + ("  [saved]" if no_improve_cnt == 0 else
                     f"  [no imp {no_improve_cnt}/{cfg['patience']}]"))

            if no_improve_cnt >= cfg["patience"]:
                print(f"  Early stopping triggered at epoch {epoch}.")
                break

        # ── Final evaluation on best checkpoint ────────────────────────────
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        _, final_acc, final_preds, final_gts = eval_epoch(model, val_loader,
                                                            criterion, device)
        fold_sec = time.time() - fold_start
        print(f"\n  Best Val Acc (Fold {fold+1}): {best_val_acc:.4f}  @ Epoch {best_epoch}"
              f"  ({fold_sec/60:.1f} min)")
        print(f"\n  Classification Report (Fold {fold+1}):\n")
        print(classification_report(final_gts, final_preds,
                                    target_names=class_names, zero_division=0))

        cm = confusion_matrix(final_gts, final_preds)
        print("  Confusion Matrix:")
        header = "           " + "  ".join(f"{c[:5]:>5}" for c in class_names)
        print(header)
        for i, row in enumerate(cm):
            print(f"  {class_names[i][:10]:<12}" +
                  "  ".join(f"{v:>5}" for v in row))

        fold_results.append(best_val_acc)

    # ── Summary ────────────────────────────────────────────────────────────
    total_min = (time.time() - run_start) / 60
    print(f"\n{'='*62}")
    print(f"  {cfg['n_folds']}-Fold CV Complete  ({total_min:.1f} min total)")
    print(f"  Fold accuracies : {[f'{a*100:.2f}%' for a in fold_results]}")
    print(f"  Mean Accuracy   : {np.mean(fold_results)*100:.2f}%")
    print(f"  Std Deviation   : {np.std(fold_results)*100:.2f}%")
    print(f"{'='*62}")
    return fold_results


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*62)
    print("  Oral Cancer CNN+BiLSTM -- Full 5-Fold 50-Epoch Run")
    print("="*62)
    print(f"  Device          : {RUN_CONFIG['device']}")
    print(f"  Epochs          : {RUN_CONFIG['epochs']}")
    print(f"  Folds           : {RUN_CONFIG['n_folds']}")
    print(f"  Batch size      : {RUN_CONFIG['batch_size']}")
    print(f"  Samples/class   : {RUN_CONFIG['n_per_class']}  ({RUN_CONFIG['n_per_class']*4} total)")
    print(f"  Early stopping  : patience={RUN_CONFIG['patience']} epochs")
    print(f"  CNN backbone    : EfficientNet-B3 (frozen, pretrained ImageNet)")
    print(f"  Clinical feats  : 13 features (base 8 + Oral Lesions, White Patches,")
    print(f"                    Chronic Sun, Poor Hygiene, Family History)")

    # 1. Images
    print("\n[1] Collecting histopathology images...")
    all_images = collect_images()

    # 2. Clinical CSV
    print("\n[2] Building balanced 4-class clinical dataset...")
    clinical_csv = os.path.join(ROOT, "data", "clinical", "merged_clinical.csv")
    cdf = build_clinical_csv(clinical_csv, RUN_CONFIG["n_per_class"])

    # 3. LSTM sequences
    print("\n[3] Encoding clinical features into LSTM sequences...")
    sequences, labels, pids, label_order = build_sequences(
        cdf, RUN_CONFIG["lstm_seq_len"]
    )
    print(f"    sequences shape : {sequences.shape}  "
          f"(patients x seq_len x features)")
    print(f"    label encoding  : {dict(enumerate(label_order))}")
    print(f"    class counts    : {dict(zip(*np.unique(labels, return_counts=True)))}")

    # 4. Image assignment
    print("\n[4] Assigning images to patients...")
    N = len(sequences)
    image_paths = assign_images(all_images, N)
    print(f"    {N} (image, sequence, label) tuples ready")
    print(f"    (images cycled from {len(all_images)} unique Mendeley slides)")

    # 5. Run full training
    print("\n[5] Starting 5-fold cross-validation  (this will take a few hours)...")
    print("    Progress is printed every epoch.  Checkpoints: ./checkpoints/\n")
    fold_results = run_full_training(
        image_paths, sequences, labels,
        RUN_CONFIG["class_names"], RUN_CONFIG
    )

    print("\n[DONE] Training complete.")
    print("  Model checkpoints saved in ./checkpoints/")
    print("  To run baseline comparison: python baseline_comparison.py")
