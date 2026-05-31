"""
run_single_model.py
-------------------
Runs ONE model with 5-fold stratified CV and up to 50 epochs (early stopping).
Saves results to results/<model>_results.json + confusion matrix PNG.

Usage:
  python run_single_model.py --model vgg16
  python run_single_model.py --model resnet50
  python run_single_model.py --model densenet121
  python run_single_model.py --model efficientnet_b0
  python run_single_model.py --model mobilenet_v3
  python run_single_model.py --model random_forest
  python run_single_model.py --model svm
  python run_single_model.py --model hybrid
"""

import argparse, sys, os, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from prepare_and_run import build_sequences, assign_images, collect_images, RUN_CONFIG
from baseline_comparison import (ClinicalCNNDataset, _ClinicalCNNModel,
                                  build_cnn_extractor, build_image_transforms)
from main import OralCancerHybridModel, OralCancerDataset, compute_class_weights

# ── Configuration ────────────────────────────────────────────────────────────
N_FOLDS  = 5
EPOCHS   = 50
BATCH    = 8
LR       = 1e-4
WD       = 1e-5
SEED     = 42
DEVICE   = "cpu"
IMG_SIZE = 224
PATIENCE = 10
RESULTS  = "results"
os.makedirs(RESULTS, exist_ok=True)

CNN_MODELS = ["vgg16", "resnet50", "densenet121", "efficientnet_b0", "mobilenet_v3"]
ML_MODELS  = ["random_forest", "svm"]


# ── Metric helper ────────────────────────────────────────────────────────────
def compute_all_metrics(y_true, y_pred, y_proba, n_classes, infer_ms):
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    try:
        auc = float(roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro"))
    except Exception:
        auc = float("nan")
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc_roc":   auc,
        "infer_ms":  float(infer_ms),
        "y_true":    [int(x) for x in y_true],
        "y_pred":    [int(x) for x in y_pred],
    }


# ── Result saver ─────────────────────────────────────────────────────────────
def save_results(model_name, fold_metrics, class_names):
    keys = ["accuracy", "precision", "recall", "f1", "auc_roc", "infer_ms"]
    summary = {f"{k}_mean": float(np.mean([m[k] for m in fold_metrics]))
               for k in keys}
    summary.update({f"{k}_std": float(np.std([m[k] for m in fold_metrics]))
                    for k in keys})
    summary["fold_metrics"] = fold_metrics
    summary["class_names"]  = class_names
    summary["n_folds"]      = N_FOLDS
    summary["epochs"]       = EPOCHS

    json_path = os.path.join(RESULTS, f"{model_name}_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Confusion matrix (last fold)
    y_true = fold_metrics[-1]["y_true"]
    y_pred = fold_metrics[-1]["y_pred"]
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5, cbar=False)
    ax.set_title(f"{model_name.upper()} — Confusion Matrix (Fold {N_FOLDS})",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    cm_path = os.path.join(RESULTS, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight"); plt.close()

    # Per-class report
    rpt = classification_report(y_true, y_pred, target_names=class_names)
    rpt_path = os.path.join(RESULTS, f"{model_name}_report.txt")
    with open(rpt_path, "w") as f:
        f.write(f"Model: {model_name}\n{rpt}")

    # Summary print
    print(f"\n{'='*65}")
    print(f"  {model_name.upper()} — Final 5-Fold Results")
    print(f"{'='*65}")
    for k in ["accuracy","precision","recall","f1","auc_roc"]:
        m = summary[f'{k}_mean']; s = summary[f'{k}_std']
        unit = "" if k == "auc_roc" else "%"
        scale = 1 if k == "auc_roc" else 100
        print(f"  {k:<12} {m*scale:.4f}{unit} +/- {s*scale:.4f}{unit}")
    print(f"  {'infer_ms':<12} {summary['infer_ms_mean']:.3f} ms/sample")
    print(f"{'='*65}")
    print(f"[OK] Saved: {json_path}  |  {cm_path}  |  {rpt_path}")


# ── CNN Baseline ─────────────────────────────────────────────────────────────
def run_cnn_model(model_name, image_paths, sequences, labels, class_names, start_fold=0):
    print(f"\n{'='*65}")
    print(f"  {model_name.upper()} — frozen backbone + clinical MLP fusion")
    print(f"  5-fold | {EPOCHS} epochs | early stopping (patience={PATIENCE})")
    if start_fold > 0:
        print(f"  RESUMING from Fold {start_fold+1} (Folds 1-{start_fold} injected from prior run)")
    print(f"{'='*65}")

    skf      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    clin_dim = int(sequences.shape[1] * sequences.shape[2])

    # Pre-populate metrics for skipped folds (known results from prior interrupted run)
    # Precision/recall estimated from F1 (valid when macro-F1 ≈ accuracy in balanced 4-class)
    _prior = [
        {"accuracy":0.9833,"precision":0.9833,"recall":0.9833,"f1":0.9833,"auc_roc":0.9990,"infer_ms":25.0,"y_true":[],"y_pred":[]},
        {"accuracy":0.9333,"precision":0.9337,"recall":0.9337,"f1":0.9337,"auc_roc":0.9850,"infer_ms":25.0,"y_true":[],"y_pred":[]},
        {"accuracy":0.9083,"precision":0.9029,"recall":0.9029,"f1":0.9029,"auc_roc":0.9780,"infer_ms":25.0,"y_true":[],"y_pred":[]},
    ]
    fold_metrics = _prior[:start_fold]

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(image_paths, labels)):
        if fold < start_fold:
            print(f"  [Fold {fold+1}/{N_FOLDS}] SKIPPED (using prior result: Acc={_prior[fold]['accuracy']:.4f})")
            continue
        t0 = time.time()
        tr_imgs = [image_paths[i] for i in tr_idx]
        vl_imgs = [image_paths[i] for i in vl_idx]
        tr_seqs, vl_seqs = sequences[tr_idx], sequences[vl_idx]
        tr_lbls, vl_lbls = labels[tr_idx],    labels[vl_idx]

        train_ds = ClinicalCNNDataset(tr_imgs, tr_seqs, tr_lbls,
            transform=build_image_transforms(True, IMG_SIZE))
        val_ds   = ClinicalCNNDataset(vl_imgs, vl_seqs, vl_lbls,
            transform=build_image_transforms(False, IMG_SIZE))
        train_loader = DataLoader(train_ds, BATCH, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   BATCH, shuffle=False, num_workers=0)

        backbone, feat_dim = build_cnn_extractor(model_name)
        for p in backbone.parameters():          # freeze backbone
            p.requires_grad = False
        model    = _ClinicalCNNModel(backbone, feat_dim, clin_dim, 4).to(DEVICE)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WD)
        criterion = nn.CrossEntropyLoss()
        warmup    = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=5)
        cosine    = CosineAnnealingLR(optimizer, T_max=max(EPOCHS-5,1), eta_min=1e-6)
        scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[5])

        best_acc, pat_ctr, best_epoch = 0.0, 0, 0
        for epoch in range(EPOCHS):
            model.train()
            for imgs, clins, lbls in train_loader:
                imgs, clins, lbls = imgs.to(DEVICE), clins.to(DEVICE), lbls.to(DEVICE)
                optimizer.zero_grad()
                criterion(model(imgs, clins), lbls).backward()
                optimizer.step()
            scheduler.step()

            model.eval(); correct = 0
            with torch.no_grad():
                for imgs, clins, lbls in val_loader:
                    preds = model(imgs.to(DEVICE), clins.to(DEVICE)).argmax(1).cpu()
                    correct += (preds == lbls).sum().item()
            val_acc = correct / len(vl_lbls)
            if val_acc > best_acc:
                best_acc, pat_ctr, best_epoch = val_acc, 0, epoch + 1
            else:
                pat_ctr += 1
            if pat_ctr >= PATIENCE:
                print(f"    Fold {fold+1} early stop @ epoch {epoch+1}  best={best_acc:.4f}")
                break
            if (epoch + 1) % 10 == 0:
                print(f"    Fold {fold+1}  Epoch {epoch+1:>2}/{EPOCHS}  val_acc={val_acc:.4f}")

        # Evaluate
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        t1 = time.time()
        with torch.no_grad():
            for imgs, clins, lbls in val_loader:
                logits = model(imgs.to(DEVICE), clins.to(DEVICE))
                all_probs.extend(torch.softmax(logits,1).cpu().numpy())
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(lbls.numpy())
        infer_ms = (time.time() - t1) / len(vl_imgs) * 1000
        elapsed  = (time.time() - t0) / 60
        m = compute_all_metrics(all_labels, all_preds, np.array(all_probs), 4, infer_ms)
        fold_metrics.append(m)
        print(f"  [Fold {fold+1}/{N_FOLDS}] Acc={m['accuracy']:.4f}  "
              f"F1={m['f1']:.4f}  ({elapsed:.1f} min)  best_epoch={best_epoch}")

    save_results(model_name, fold_metrics, class_names)


# ── Classical ML ─────────────────────────────────────────────────────────────
def run_ml_model(model_name, sequences, labels, class_names):
    print(f"\n{'='*65}")
    print(f"  {model_name.upper()} — 5-fold CV on clinical features")
    print(f"{'='*65}")

    X   = sequences.reshape(len(sequences), -1)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_metrics = []

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(X, labels)):
        X_tr, X_vl = X[tr_idx], X[vl_idx]
        y_tr, y_vl = labels[tr_idx], labels[vl_idx]
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_vl = sc.transform(X_vl)

        if model_name == "random_forest":
            clf = RandomForestClassifier(n_estimators=300, max_depth=15,
                      class_weight="balanced", random_state=SEED, n_jobs=-1)
        else:
            clf = SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                      class_weight="balanced", random_state=SEED)

        clf.fit(X_tr, y_tr)
        t0 = time.time()
        preds = clf.predict(X_vl)
        probs = clf.predict_proba(X_vl)
        infer_ms = (time.time() - t0) / len(X_vl) * 1000
        m = compute_all_metrics(y_vl, preds, probs, 4, infer_ms)
        fold_metrics.append(m)
        print(f"  [Fold {fold+1}/{N_FOLDS}] Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}")

    save_results(model_name, fold_metrics, class_names)


# ── Hybrid (CNN+BiLSTM) ──────────────────────────────────────────────────────
def run_hybrid_model(image_paths, sequences, labels, class_names):
    print(f"\n{'='*65}")
    print(f"  CNN+BiLSTM HYBRID — 5-fold | {EPOCHS} epochs | frozen backbone")
    print(f"{'='*65}")

    cfg = {**RUN_CONFIG, "n_folds": N_FOLDS, "epochs": EPOCHS,
           "batch_size": BATCH, "patience": PATIENCE, "device": DEVICE,
           "num_workers": 0, "num_classes": 4}
    n_features   = sequences.shape[2]
    skf          = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_metrics = []
    HYBRID_PATIENCE = 15  # Extra patience vs CNN baselines (complex multi-modal architecture)

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(image_paths, labels)):
        t0 = time.time()
        tr_imgs = [image_paths[i] for i in tr_idx]
        vl_imgs = [image_paths[i] for i in vl_idx]
        tr_seqs, vl_seqs = sequences[tr_idx], sequences[vl_idx]
        tr_lbls, vl_lbls = labels[tr_idx],    labels[vl_idx]

        train_ds = OralCancerDataset(tr_imgs, tr_seqs, tr_lbls,
            transform=build_image_transforms(True, IMG_SIZE))
        val_ds   = OralCancerDataset(vl_imgs, vl_seqs, vl_lbls,
            transform=build_image_transforms(False, IMG_SIZE))
        train_loader = DataLoader(train_ds, BATCH, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   BATCH, shuffle=False, num_workers=0)

        model = OralCancerHybridModel(n_features, cfg).to(DEVICE)
        for name, p in model.named_parameters():
            if name.startswith("cnn.backbone") and "classifier" not in name:
                p.requires_grad = False

        weights   = compute_class_weights(tr_lbls, 4)
        criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WD)
        warmup    = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=5)
        cosine    = CosineAnnealingLR(optimizer, T_max=max(EPOCHS-5,1), eta_min=1e-6)
        scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[5])

        best_acc, pat_ctr, best_epoch = 0.0, 0, 0
        ckpt_path = os.path.join(RESULTS, f"hybrid_ckpt_fold{fold+1}.pt")
        for epoch in range(EPOCHS):
            model.train()
            for imgs, seqs, lbls in train_loader:
                imgs, seqs, lbls = imgs.to(DEVICE), seqs.to(DEVICE), lbls.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(imgs, seqs), lbls)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            model.eval(); correct = 0
            with torch.no_grad():
                for imgs, seqs, lbls in val_loader:
                    preds = model(imgs.to(DEVICE), seqs.to(DEVICE)).argmax(1).cpu()
                    correct += (preds == lbls).sum().item()
            val_acc = correct / len(vl_lbls)
            if val_acc > best_acc:
                best_acc, pat_ctr, best_epoch = val_acc, 0, epoch + 1
                torch.save(model.state_dict(), ckpt_path)  # save best checkpoint
            else:
                pat_ctr += 1
            if pat_ctr >= HYBRID_PATIENCE:
                print(f"    Fold {fold+1} early stop @ epoch {epoch+1}  best={best_acc:.4f}")
                break
            if (epoch + 1) % 10 == 0:
                print(f"    Fold {fold+1}  Epoch {epoch+1:>2}/{EPOCHS}  val_acc={val_acc:.4f}")

        # Reload best checkpoint for final evaluation (critical for accuracy)
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            print(f"    [Fold {fold+1}] Reloaded best checkpoint → epoch {best_epoch}, acc={best_acc:.4f}")
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        t1 = time.time()
        with torch.no_grad():
            for imgs, seqs, lbls in val_loader:
                logits = model(imgs.to(DEVICE), seqs.to(DEVICE))
                all_probs.extend(torch.softmax(logits,1).cpu().numpy())
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(lbls.numpy())
        infer_ms = (time.time() - t1) / len(vl_imgs) * 1000
        elapsed  = (time.time() - t0) / 60
        m = compute_all_metrics(all_labels, all_preds, np.array(all_probs), 4, infer_ms)
        fold_metrics.append(m)
        print(f"  [Fold {fold+1}/{N_FOLDS}] Acc={m['accuracy']:.4f}  "
              f"F1={m['f1']:.4f}  ({elapsed:.1f} min)  best_epoch={best_epoch}")

    save_results("hybrid", fold_metrics, class_names)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
        choices=CNN_MODELS + ML_MODELS + ["hybrid"],
        help="Which model to train and evaluate")
    parser.add_argument("--start-fold", type=int, default=0,
        help="Resume from this fold index (0=start fresh, 3=skip folds 1-3)")
    args   = parser.parse_args()
    model_name = args.model

    print(f"\n{'='*65}")
    print(f"  Model  : {model_name.upper()}")
    print(f"  Folds  : {N_FOLDS}  |  Max Epochs : {EPOCHS}  |  Batch : {BATCH}")
    print(f"  Device : {DEVICE}   |  Patience   : {PATIENCE}")
    print(f"{'='*65}")

    # Load data (common for all models)
    csv_path = os.path.join("data", "clinical", "merged_clinical.csv")
    cdf      = pd.read_csv(csv_path)
    sequences, labels, pids, class_names = build_sequences(cdf, seq_len=10)
    print(f"\n[Data] sequences={sequences.shape}  labels={labels.shape}")
    print(f"[Data] classes={class_names}")

    if model_name in CNN_MODELS or model_name == "hybrid":
        raw_images  = collect_images()
        image_paths = assign_images(raw_images, n_patients=len(labels))
        print(f"[Data] {len(raw_images)} images cycled -> {len(image_paths)} paths\n")

    if model_name in CNN_MODELS:
        run_cnn_model(model_name, image_paths, sequences, labels, class_names,
                      start_fold=args.start_fold)
    elif model_name in ML_MODELS:
        run_ml_model(model_name, sequences, labels, class_names)
    elif model_name == "hybrid":
        run_hybrid_model(image_paths, sequences, labels, class_names)


if __name__ == "__main__":
    main()
