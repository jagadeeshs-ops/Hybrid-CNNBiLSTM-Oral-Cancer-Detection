"""
run_baseline.py -- Adapter that feeds correctly-prepared data into baseline_comparison.py
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

import glob, itertools, pandas as pd, numpy as np
from prepare_and_run import build_sequences, assign_images, collect_images
from baseline_comparison import (
    COMPARE_CONFIG, run_full_comparison,
    plot_accuracy_comparison, plot_metrics_radar,
    plot_confusion_matrices, plot_inference_time,
    plot_f1_per_class, generate_summary_table,
)

RUN_CFG = {
    **COMPARE_CONFIG,
    "batch_size":      8,
    "num_workers":     0,
    "baseline_epochs": 3,
    "device":          "cpu",
}

RESULTS_DIR = RUN_CFG["results_dir"]
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    print("\n" + "=" * 65)
    print("  Oral Cancer Detection -- Baseline Comparison Suite")
    print("=" * 65)

    # 1. Load clinical CSV
    csv_path = os.path.join("data", "clinical", "merged_clinical.csv")
    print(f"\n[1] Loading clinical CSV: {csv_path}")
    cdf = pd.read_csv(csv_path)
    print(f"    {len(cdf)} rows, {len(cdf.columns)} columns")

    sequences, labels, pids, class_names = build_sequences(
        cdf, seq_len=RUN_CFG["lstm_seq_len"]
    )
    print(f"    Sequences: {sequences.shape}  Labels: {labels.shape}")
    print(f"    Classes: {class_names}")
    RUN_CFG["class_names"] = class_names

    # 2. Collect & cycle images
    print("\n[2] Collecting images...")
    raw_images = collect_images()
    if not raw_images:
        sys.exit("ERROR: No images found under data/Mendeley/training/")
    image_paths = assign_images(raw_images, n_patients=len(labels))
    print(f"    {len(raw_images)} raw images cycled -> {len(image_paths)} paths")

    assert len(image_paths) == len(sequences) == len(labels)

    # 3. Run all models
    print(f"\n[3] Running comparison "
          f"(epochs={RUN_CFG['baseline_epochs']}, "
          f"batch={RUN_CFG['batch_size']}, n_folds=3) ...")
    results = run_full_comparison(image_paths, sequences, labels, RUN_CFG)

    # 4. Plots
    print("\n[4] Generating visualizations...")
    rd = RESULTS_DIR
    plot_accuracy_comparison(results, f"{rd}/accuracy_comparison.png")
    plot_metrics_radar(results,       f"{rd}/radar_comparison.png")
    plot_confusion_matrices(results, class_names, f"{rd}/confusion_matrices_baselines.png")
    plot_inference_time(results,      f"{rd}/inference_time.png")
    plot_f1_per_class(results, class_names, f"{rd}/f1_per_class.png")

    # 5. Summary table
    print("\n[5] Generating summary table...")
    generate_summary_table(results, f"{rd}/model_comparison_table.html")

    print(f"\n[OK] All results saved to ./{rd}/")


if __name__ == "__main__":
    main()
