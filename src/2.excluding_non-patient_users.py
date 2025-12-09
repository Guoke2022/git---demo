# ------------------------------------------------------------------------------
# 2.excluding_non-patient_users.py
#
# This script performs a preprocessing step for the synthetic mobile trajectory
# dataset used in this project. It identifies and removes hospital staff users
# from daily trajectory records based on their long-term appearance frequency.
#
# The workflow includes:
#   1. Loading multiple daily single_day_patients_*.csv files.
#   2. Identifying hospital staff as users appearing on ≥ N days.
#   3. Saving a staff-ID list to disk.
#   4. Generating cleaned daily trajectory files with staff removed.
#
# This script is designed for demonstration only.
# It operates on fully synthetic data without any real or personal information.
# Therefore, results are NOT intended for real-world inference or validation.
# ------------------------------------------------------------------------------
 

import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)


def identify_hospital_staff(input_path: Path, worker_threshold: int) -> pd.DataFrame:
    """
    Identify hospital staff users based on appearance frequency across multiple days.

    Parameters
    ----------
    input_path : Path
        Folder containing daily `single_day_patients_*.csv` files.
    worker_threshold : int
        Users appearing in ≥ worker_threshold days are classified as hospital staff.

    Returns
    -------
    pd.DataFrame
        DataFrame containing IDs of identified hospital staff.
    """
    print("Loading daily single_day_patients files...")

    all_files = sorted(input_path.glob("single_day_patients_*.csv"))
    dfs = []

    for f in all_files:
        df = pd.read_csv(f)
        df["day_file"] = f.name
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)

    # Count distinct days per user
    id_day_counts = full_df.groupby("id")["day_file"].nunique()

    # Staff appear in >= threshold days
    worker_ids = id_day_counts[id_day_counts >= worker_threshold].index
    worker_df = pd.DataFrame({"id": worker_ids})

    print(f"Identified {len(worker_df)} hospital staff users.")
    return worker_df


def remove_staff_from_daily_files(input_path: Path, output_path: Path, staff_ids: pd.Series):
    """
    Remove hospital staff from all daily trajectory files and save cleaned versions.

    Parameters
    ----------
    input_path : Path
        Folder containing input daily CSV files.
    output_path : Path
        Folder where cleaned CSV files will be saved.
    staff_ids : pd.Series
        Series containing all staff user IDs.
    """
    print("\nFiltering out hospital staff from each file...\n")

    all_files = sorted(input_path.glob("single_day_patients_*.csv"))

    for f in all_files:
        df = pd.read_csv(f)
        before = len(df)

        # Filter out staff
        df_clean = df[~df["id"].isin(staff_ids)]
        after = len(df_clean)

        # Output filename
        new_name = f.stem + "_drop_staff.csv"
        out_path = output_path / new_name

        df_clean.to_csv(out_path, index=False, encoding="utf-8-sig")

        print(f"{f.name}: {before} → {after} (removed {before - after}) → saved as {new_name}")

    print("\nAll files processed successfully!")


def main():
    # --------------------------
    # Parameter settings
    # --------------------------
    INPUT_PATH = Path("../data/result_test")
    OUTPUT_PATH = Path("../data/result_test/drop_staff")
    WORKER_THRESHOLD = 20  # Appear ≥20 days → hospital staff

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Step 1 — Identify hospital staff
    staff_df = identify_hospital_staff(INPUT_PATH, WORKER_THRESHOLD)

    # Save staff list
    staff_list_path = OUTPUT_PATH / "hospital_staff_list.csv"
    staff_df.to_csv(staff_list_path, index=False, encoding="utf-8-sig")
    print(f"\nHospital staff list saved: {staff_list_path}")

    # Step 2 — Remove staff from each file
    remove_staff_from_daily_files(INPUT_PATH, OUTPUT_PATH, staff_df["id"])


if __name__ == "__main__":
    main()