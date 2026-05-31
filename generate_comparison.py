"""
generate_comparison.py
----------------------
Loads all results/<model>_results.json files produced by run_single_model.py
and generates a comprehensive comparison table + publication-quality charts.

Usage:
  python generate_comparison.py
"""

import sys, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

RESULTS_DIR = "results"

# Ordered display names
MODEL_ORDER = [
    "random_forest",
    "svm",
    "vgg16",
    "resnet50",
    "densenet121",
    "efficientnet_b0",
    "mobilenet_v3",
    "hybrid",
]

DISPLAY_NAMES = {
    "random_forest":   "Random Forest",
    "svm":             "SVM (RBF)",
    "vgg16":           "VGG16 + Clinical",
    "resnet50":        "ResNet50 + Clinical",
    "densenet121":     "DenseNet121 + Clinical",
    "efficientnet_b0": "EfficientNet-B0 + Clinical",
    "mobilenet_v3":    "MobileNetV3 + Clinical",
    "hybrid":          "CNN+BiLSTM (Proposed)",
}

METRICS = ["accuracy", "precision", "recall", "f1", "auc_roc", "infer_ms"]
METRIC_LABELS = {
    "accuracy":  "Accuracy (%)",
    "precision": "Precision (%)",
    "recall":    "Recall (%)",
    "f1":        "F1-Score (%)",
    "auc_roc":   "AUC-ROC",
    "infer_ms":  "Inference (ms/sample)",
}


# ── Load all JSON results ─────────────────────────────────────────────────────
def load_all_results():
    rows = []
    missing = []
    for m in MODEL_ORDER:
        path = os.path.join(RESULTS_DIR, f"{m}_results.json")
        if not os.path.exists(path):
            missing.append(m)
            continue
        with open(path) as f:
            data = json.load(f)
        row = {"model": m, "display": DISPLAY_NAMES[m]}
        for k in METRICS:
            row[f"{k}_mean"] = data.get(f"{k}_mean", float("nan"))
            row[f"{k}_std"]  = data.get(f"{k}_std",  float("nan"))
        rows.append(row)
    if missing:
        print(f"[WARN] Missing result files for: {missing}")
        print(f"       Run: python run_single_model.py --model <name> for each missing model\n")
    return pd.DataFrame(rows)


# ── Print comparison table ────────────────────────────────────────────────────
def print_table(df):
    print("\n" + "="*95)
    print(f"  {'Model':<32} {'Acc%':>8} {'Prec%':>8} {'Rec%':>8} {'F1%':>8} {'AUC-ROC':>8} {'ms/smp':>8}")
    print("="*95)
    for _, row in df.iterrows():
        tag = " <-- Proposed" if row["model"] == "hybrid" else ""
        print(
            f"  {row['display']:<32}"
            f"  {row['accuracy_mean']*100:>6.2f}"
            f"  {row['precision_mean']*100:>7.2f}"
            f"  {row['recall_mean']*100:>6.2f}"
            f"  {row['f1_mean']*100:>6.2f}"
            f"  {row['auc_roc_mean']:>7.4f}"
            f"  {row['infer_ms_mean']:>7.3f}{tag}"
        )
    print("="*95)


# ── Save CSV table ────────────────────────────────────────────────────────────
def save_csv(df):
    out = []
    for _, row in df.iterrows():
        r = {"Model": row["display"]}
        for k in METRICS:
            if k == "infer_ms":
                r["Inference (ms)"] = f"{row[f'{k}_mean']:.3f} +/- {row[f'{k}_std']:.3f}"
            elif k == "auc_roc":
                r["AUC-ROC"] = f"{row[f'{k}_mean']:.4f} +/- {row[f'{k}_std']:.4f}"
            else:
                label = k.capitalize()
                r[label] = (f"{row[f'{k}_mean']*100:.2f}% "
                            f"+/- {row[f'{k}_std']*100:.2f}%")
        out.append(r)
    cdf = pd.DataFrame(out)
    csv_path = os.path.join(RESULTS_DIR, "comparison_table.csv")
    cdf.to_csv(csv_path, index=False)
    print(f"[OK] Saved: {csv_path}")
    return cdf


# ── Bar chart helper ──────────────────────────────────────────────────────────
COLORS = [
    "#4878CF", "#4878CF", "#6ACC65", "#6ACC65",
    "#6ACC65", "#6ACC65", "#6ACC65", "#D65F5F",
]

def bar_chart(df, metric, out_path, title):
    vals   = df[f"{metric}_mean"].values
    errs   = df[f"{metric}_std"].values
    names  = df["display"].tolist()
    scale  = 100 if metric not in ("auc_roc", "infer_ms") else 1
    vals   = vals * scale
    errs   = errs * scale

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(range(len(df)), vals, yerr=errs, capsize=4,
                  color=COLORS[:len(df)], edgecolor="white",
                  linewidth=0.8, error_kw={"elinewidth": 1.2, "ecolor": "#333"})

    # Highlight proposed model
    for i, m in enumerate(df["model"].tolist()):
        if m == "hybrid":
            bars[i].set_edgecolor("#B22222")
            bars[i].set_linewidth(2.5)

    # Value labels on bars
    for i, (v, e) in enumerate(zip(vals, errs)):
        ax.text(i, v + e + 0.3, f"{v:.2f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ylabel = METRIC_LABELS.get(metric, metric)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)

    legend_patches = [
        mpatches.Patch(color="#4878CF", label="Classical ML"),
        mpatches.Patch(color="#6ACC65", label="CNN+Clinical Fusion"),
        mpatches.Patch(color="#D65F5F", label="Proposed CNN+BiLSTM"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {out_path}")


# ── Multi-metric grouped bar chart ────────────────────────────────────────────
def grouped_bar_chart(df):
    metrics  = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    n_models = len(df)
    n_met    = len(metrics)
    x        = np.arange(n_models)
    width    = 0.15
    palette  = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#937860"]

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (met, col) in enumerate(zip(metrics, palette)):
        vals = df[f"{met}_mean"].values * 100
        errs = df[f"{met}_std"].values  * 100
        offset = (i - n_met / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, yerr=errs, label=met.upper(),
               color=col, alpha=0.85, capsize=3,
               error_kw={"elinewidth": 1, "ecolor": "#333"})

    ax.set_xticks(x)
    ax.set_xticklabels(df["display"].tolist(), rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("All Metrics Comparison — All Models", fontsize=13,
                 fontweight="bold", pad=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4); ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9, ncol=3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "comparison_grouped_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] Saved: {path}")


# ── Heatmap of all metrics ────────────────────────────────────────────────────
def metrics_heatmap(df):
    cols   = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    labels = [METRIC_LABELS[c] for c in cols]
    data   = df[[f"{c}_mean" for c in cols]].values * 100
    idx    = df["display"].tolist()

    fig, ax = plt.subplots(figsize=(9, max(4, len(idx) * 0.55)))
    sns.heatmap(data, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=labels, yticklabels=idx, ax=ax,
                linewidths=0.5, cbar_kws={"label": "Score (%)"})
    ax.set_title("Model Comparison Heatmap", fontsize=13, fontweight="bold", pad=10)
    plt.xticks(rotation=20, ha="right", fontsize=9)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "comparison_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] Saved: {path}")


# ── LaTeX table ───────────────────────────────────────────────────────────────
def save_latex(df):
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Performance Comparison of Baseline Models vs. Proposed CNN+BiLSTM Framework}")
    lines.append(r"\label{tab:comparison}")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Acc (\%)} & \textbf{Prec (\%)} & "
                 r"\textbf{Rec (\%)} & \textbf{F1 (\%)} & \textbf{AUC-ROC} & "
                 r"\textbf{ms/smp} \\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        name = row["display"].replace("+", r"\texttt{+}")
        if row["model"] == "hybrid":
            name = r"\textbf{" + name + r"}"
        acc  = f"{row['accuracy_mean']*100:.2f} $\\pm$ {row['accuracy_std']*100:.2f}"
        prec = f"{row['precision_mean']*100:.2f} $\\pm$ {row['precision_std']*100:.2f}"
        rec  = f"{row['recall_mean']*100:.2f} $\\pm$ {row['recall_std']*100:.2f}"
        f1   = f"{row['f1_mean']*100:.2f} $\\pm$ {row['f1_std']*100:.2f}"
        auc  = f"{row['auc_roc_mean']:.4f} $\\pm$ {row['auc_roc_std']:.4f}"
        ms   = f"{row['infer_ms_mean']:.3f}"
        lines.append(f"{name} & {acc} & {prec} & {rec} & {f1} & {auc} & {ms} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    tex = "\n".join(lines)

    tex_path = os.path.join(RESULTS_DIR, "comparison_table.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"[OK] Saved: {tex_path}")
    print("\n--- LaTeX Table (copy-paste ready) ---")
    print(tex)
    return tex


# ── Improvement analysis ──────────────────────────────────────────────────────
def improvement_analysis(df):
    if "hybrid" not in df["model"].values:
        print("[SKIP] No hybrid results yet.")
        return
    hyb = df[df["model"] == "hybrid"].iloc[0]
    others = df[df["model"] != "hybrid"]

    print("\n" + "="*75)
    print("  Improvement of CNN+BiLSTM over Baselines")
    print("="*75)
    print(f"  {'Model':<32} {'dAcc%':>7} {'dF1%':>7} {'dAUC':>7}")
    print("-"*75)
    for _, row in others.iterrows():
        da = (hyb["accuracy_mean"] - row["accuracy_mean"]) * 100
        df1 = (hyb["f1_mean"] - row["f1_mean"]) * 100
        dauc = hyb["auc_roc_mean"] - row["auc_roc_mean"]
        print(f"  {row['display']:<32} {da:>+7.2f} {df1:>+7.2f} {dauc:>+7.4f}")
    print("="*75)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*65)
    print("  Oral Cancer Detection -- Final Comparison Report")
    print("="*65)

    df = load_all_results()
    if df.empty:
        print("[ERROR] No result files found in results/. Run models first.")
        return

    print(f"\n[INFO] Loaded results for {len(df)} model(s): "
          f"{df['display'].tolist()}")

    print_table(df)
    save_csv(df)

    # Charts
    bar_chart(df, "accuracy",
              os.path.join(RESULTS_DIR, "comparison_accuracy_bar.png"),
              "Accuracy Comparison — All Models (5-fold CV)")
    bar_chart(df, "f1",
              os.path.join(RESULTS_DIR, "comparison_f1_bar.png"),
              "F1-Score Comparison — All Models (5-fold CV)")
    bar_chart(df, "auc_roc",
              os.path.join(RESULTS_DIR, "comparison_auc_bar.png"),
              "AUC-ROC Comparison — All Models (5-fold CV)")
    bar_chart(df, "infer_ms",
              os.path.join(RESULTS_DIR, "comparison_inference_bar.png"),
              "Inference Time (ms/sample) — All Models")

    if len(df) > 1:
        grouped_bar_chart(df)
    metrics_heatmap(df)
    save_latex(df)
    improvement_analysis(df)

    print("\n[DONE] All comparison files saved to:", os.path.abspath(RESULTS_DIR))


if __name__ == "__main__":
    main()
