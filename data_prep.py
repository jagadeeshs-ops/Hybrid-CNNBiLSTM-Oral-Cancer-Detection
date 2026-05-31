"""
data_prep.py — Dataset Merging & Preparation Utility
======================================================
Merges Mendeley, NDB-UFES, Kaggle, and TCGA clinical data
into unified image manifest + clinical CSV for the pipeline.

Run this ONCE before main.py to prepare your data.
"""

import os
import shutil
import pandas as pd
import numpy as np
from glob import glob
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────
#  STEP 1: Build Unified Image Manifest
# ─────────────────────────────────────────────

def build_image_manifest(
    mendeley_dir: str,
    ndbufes_dir: str,
    output_csv: str = "data/image_manifest.csv"
):
    """
    Walks Mendeley and NDB-UFES image directories.
    Assumes subfolders are named by class label, e.g.:
      mendeley/
        OSCC/img001.jpg
        Normal/img002.jpg
      ndbufes/
        Leukoplakia_Dysplasia/img003.png
        Leukoplakia_NoDysplasia/img004.png
        OSCC/img005.png

    Returns a DataFrame: [image_path, label, source]
    """
    records = []

    for base_dir, source_name in [(mendeley_dir, "mendeley"),
                                   (ndbufes_dir,  "ndbufes")]:
        if not os.path.exists(base_dir):
            print(f"[WARN] Directory not found: {base_dir}")
            continue
        for class_folder in os.listdir(base_dir):
            class_path = os.path.join(base_dir, class_folder)
            if not os.path.isdir(class_path):
                continue
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.tiff"]:
                for img_path in glob(os.path.join(class_path, ext)):
                    records.append({
                        "image_path": img_path,
                        "label":      class_folder,   # folder name = class
                        "source":     source_name,
                    })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[✓] Image manifest saved: {output_csv}")
    print(f"    Total images: {len(df)}")
    print(df["label"].value_counts().to_string())
    return df


# ─────────────────────────────────────────────
#  STEP 2: Merge Clinical CSVs
# ─────────────────────────────────────────────

def merge_clinical_data(
    ndbufes_csv: str,
    kaggle_csv: str,
    tcga_csv: str = None,
    output_csv: str = "data/clinical/merged_clinical.csv"
):
    """
    Merges NDB-UFES patient data, Kaggle oral cancer CSV,
    and optionally TCGA clinical metadata into a single CSV.

    Expected column mappings (adapt to actual column names):
    ─────────────────────────────────────────────────────────
    NDB-UFES  : patient_id, age, gender, lesion_type, visit_date, label
    Kaggle    : patient_id, age, gender, tobacco_use, alcohol_use,
                betel_nut, hpv_status, ulcer_size, pain_score, label
    TCGA      : case_id, age_at_diagnosis, gender, tobacco_smoking_history,
                alcohol_history, hpv_status, days_to_last_follow_up, label

    All sources are harmonized to a common schema.
    """
    dfs = []

    # --- NDB-UFES ---
    if ndbufes_csv and os.path.exists(ndbufes_csv):
        df_ndb = pd.read_csv(ndbufes_csv)
        # Rename to standard schema (adapt as needed)
        df_ndb = df_ndb.rename(columns={
            "case_id":     "patient_id",
            "lesion_type": "label",
        })
        df_ndb["source"] = "ndbufes"
        # Add missing columns with NaN
        for col in ["tobacco_use", "alcohol_use", "betel_nut",
                    "hpv_status", "ulcer_size", "pain_score", "visit_month"]:
            if col not in df_ndb.columns:
                df_ndb[col] = np.nan
        dfs.append(df_ndb)
        print(f"[✓] NDB-UFES loaded: {len(df_ndb)} records")

    # --- Kaggle CSV ---
    if kaggle_csv and os.path.exists(kaggle_csv):
        df_kag = pd.read_csv(kaggle_csv)
        df_kag["source"] = "kaggle"
        if "visit_month" not in df_kag.columns:
            df_kag["visit_month"] = 0  # single visit
        dfs.append(df_kag)
        print(f"[✓] Kaggle loaded: {len(df_kag)} records")

    # --- TCGA (optional) ---
    if tcga_csv and os.path.exists(tcga_csv):
        df_tcga = pd.read_csv(tcga_csv)
        df_tcga = df_tcga.rename(columns={
            "case_id":                  "patient_id",
            "age_at_diagnosis":         "age",
            "tobacco_smoking_history":  "tobacco_use",
            "alcohol_history":          "alcohol_use",
            "days_to_last_follow_up":   "visit_month",
        })
        df_tcga["source"]    = "tcga"
        df_tcga["label"]     = df_tcga.get("primary_diagnosis", "OSCC")
        df_tcga["betel_nut"] = np.nan
        dfs.append(df_tcga)
        print(f"[✓] TCGA loaded: {len(df_tcga)} records")

    if not dfs:
        raise FileNotFoundError("No clinical CSV files found. Check your paths.")

    merged = pd.concat(dfs, ignore_index=True)

    # --- Standardize label names ---
    label_map = {
        "carcinoma":                   "OSCC",
        "oscc":                        "OSCC",
        "squamous cell carcinoma":     "OSCC",
        "leukoplakia with dysplasia":  "Leukoplakia_Dysplasia",
        "leukoplakia_dysplasia":       "Leukoplakia_Dysplasia",
        "leukoplakia without dysplasia": "Leukoplakia_NoDysplasia",
        "leukoplakia_nodysplasia":     "Leukoplakia_NoDysplasia",
        "normal":                      "Normal",
        "benign":                      "Normal",
    }
    merged["label"] = (merged["label"]
                       .str.lower()
                       .str.strip()
                       .map(lambda x: label_map.get(x, x)))

    # --- Fill missing values ---
    num_cols = ["age", "ulcer_size", "pain_score", "visit_month"]
    for col in num_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(merged[col].median())

    cat_cols = ["gender", "tobacco_use", "alcohol_use",
                "betel_nut", "hpv_status"]
    for col in cat_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna("unknown")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    merged.to_csv(output_csv, index=False)
    print(f"\n[✓] Merged clinical CSV saved: {output_csv}")
    print(f"    Total records : {len(merged)}")
    print(f"    Label counts  :\n{merged['label'].value_counts().to_string()}")
    return merged


# ─────────────────────────────────────────────
#  STEP 3: Validate Image-CSV Alignment
# ─────────────────────────────────────────────

def validate_alignment(image_manifest_csv: str, clinical_csv: str):
    """
    Checks that patient IDs in the clinical CSV can be linked
    to images in the manifest. Prints a diagnostic report.

    In your dataset, you may need to:
      - Extract patient_id from image filenames (e.g., "P001_slide1.jpg" → P001)
      - Match to patient_id column in clinical CSV
    """
    img_df = pd.read_csv(image_manifest_csv)
    clin_df = pd.read_csv(clinical_csv)

    # Extract patient IDs from image filenames
    img_df["patient_id"] = (img_df["image_path"]
                             .apply(lambda p: os.path.basename(p).split("_")[0]))

    img_pids  = set(img_df["patient_id"].unique())
    clin_pids = set(clin_df["patient_id"].astype(str).unique())

    matched   = img_pids & clin_pids
    img_only  = img_pids - clin_pids
    clin_only = clin_pids - img_pids

    print(f"\n[Alignment Report]")
    print(f"  Images with matching clinical record : {len(matched)}")
    print(f"  Images WITHOUT clinical record       : {len(img_only)}")
    print(f"  Clinical records WITHOUT images      : {len(clin_only)}")

    if len(img_only) > 0:
        print(f"  [WARN] {len(img_only)} images have no clinical match — "
              f"will use zero-padded sequences for LSTM stream.")
    return matched, img_only, clin_only


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("="*55)
    print("  Oral Cancer Dataset Preparation")
    print("="*55)

    # Step 1: Build image manifest
    img_manifest = build_image_manifest(
        mendeley_dir="data/mendeley/images",
        ndbufes_dir="data/ndbufes/images",
        output_csv="data/image_manifest.csv",
    )

    # Step 2: Merge clinical CSVs
    clinical = merge_clinical_data(
        ndbufes_csv="data/ndbufes/patient_data.csv",
        kaggle_csv="data/kaggle/oral_cancer_prediction.csv",
        tcga_csv="data/tcga/clinical.csv",          # set None if not using TCGA
        output_csv="data/clinical/merged_clinical.csv",
    )

    # Step 3: Validate alignment
    validate_alignment(
        image_manifest_csv="data/image_manifest.csv",
        clinical_csv="data/clinical/merged_clinical.csv",
    )

    print("\n[✓] Data preparation complete. Run main.py to start training.")
