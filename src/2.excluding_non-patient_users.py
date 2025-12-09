# ------------------------------------------------------------------------------
# This script processes synthetic mobile trajectory data used only for demonstration.
# No real or personal data is included. Results generated from this dataset
# SHOULD NOT be interpreted as real-world findings or algorithmic validation.
# ------------------------------------------------------------------------------
 

import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)


# --------------------------
# Parameter settings
# --------------------------


INPUT_PATH = Path("../data/result_test")         # Folder containing 60 daily single_day_patients files
OUTPUT_PATH = Path("../data/result_test/drop_staff")      # Folder to save cleaned files without hospital staff
WORKER_THRESHOLD = 20                            # If a user appears ≥20 days, classify as hospital staff

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# ==============================================================
# 1. Read all daily files and identify hospital staff users
# ==============================================================

print("Loading all single_day_patients files...")

all_files = sorted(INPUT_PATH.glob("single_day_patients_*.csv"))
dfs = []

for f in all_files:
    df = pd.read_csv(f)
    df["day_file"] = f.name  # Track which day the record comes from
    dfs.append(df)

full_df = pd.concat(dfs, ignore_index=True)

# Count how many distinct days each user appears
id_day_counts = full_df.groupby("id")["day_file"].nunique()

# Identify hospital staff (appear in ≥ WORKER_THRESHOLD days)
worker_ids = id_day_counts[id_day_counts >= WORKER_THRESHOLD].index
worker_df = pd.DataFrame({"id": worker_ids})

# Save staff list
worker_list_path = OUTPUT_PATH / f"hospital_staff_list.csv"
worker_df.to_csv(worker_list_path, index=False, encoding="utf-8-sig")
print(f"Hospital staff list saved: {worker_list_path}")

# ==============================================================
# 2. Remove hospital staff from each daily file and save cleaned files
# ==============================================================

print("\nFiltering out hospital staff from each daily file...\n")

for f in all_files:
    df = pd.read_csv(f)
    before = len(df)

    # Remove staff IDs
    df_filtered = df[~df["id"].isin(worker_ids)]
    after = len(df_filtered)

    new_name = f.stem + "_drop_staff.csv"
    out_path = OUTPUT_PATH / new_name
    df_filtered.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"{f.name}: {before} → {after} after removing staff → saved as {new_name}")

print("\nProcessing completed successfully!")