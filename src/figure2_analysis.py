# ------------------------------------------------------------------------------
# figure2_analysis.py
# 
# Purpose:
#   - Generate summary CSV files for NNHI, extra travel distance, and bypass rate
#     including subsets for top-tier hospitals and high-reputation hospitals
#   - Generate scatter plots with log-fitted line and bar plots for each measure
# ------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pathlib2 import Path
import warnings

warnings.filterwarnings('ignore')  # Ignore all warnings
pd.set_option('display.max_columns', None)

plt.rcParams['font.family'] = 'Arial'

# ======================================================
# Paths (use relative paths for GitHub)
# ======================================================
DATA_PATH = Path("./data")  # input CSV folder
OUTPUT_PATH = Path("./output/Figure2")  # output folder
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Hospital info CSV
HOSP_PATH = DATA_PATH / "hospital_info.csv"  # should contain columns: name, top_tier, high_reputation

# Main patient data CSV
PATIENT_CSV = DATA_PATH / "patient_summary.csv"  # should contain columns: id, name, HousePrice, NNHI, home_road_dist, road_closest

# ======================================================
# Read main patient data
# ======================================================
df = pd.read_csv(PATIENT_CSV, encoding="utf-8")
df = df[['id', 'name', 'HousePrice', 'NNHI', 'home_road_dist', 'road_closest']]

# Compute extra travel distance
df["extra_distance"] = df["home_road_dist"] - df["road_closest"]

# Bin extra distance (0.5 km bins)
bins = np.arange(0, df["extra_distance"].max() + 0.5, 0.5)
labels = bins[1:]
df["extra_distance_group"] = pd.cut(df["extra_distance"], bins=bins, labels=labels, right=False).astype(float)

# ======================================================
# Read hospital info and merge
# ======================================================
hosp = pd.read_csv(HOSP_PATH, encoding="utf-8")
hosp = hosp[['name', 'top_tier', 'high_reputation']]

df = df.merge(hosp, on='name', how='left')

# Create subsets
df_top = df[df["top_tier"] == 1]
df_high = df[df["high_reputation"] == 1]

# Total unique patients
total_n = df['id'].nunique()

# ======================================================
# Function to generate summary CSVs
# ======================================================
def generate_summary_csv(df_subset, subset_name):
    # -------------------------
    # 1) bypass rate by HousePrice
    # -------------------------
    group_stats = (
        df.groupby('HousePrice')
          .agg(group_size=('id', 'nunique'),
               bypass_rate=('NNHI', lambda x: (x > 1).mean()))
          .reset_index()
    )

    group_stats_subset = (
        df_subset.groupby('HousePrice')
                 .agg(group_size_subset=('id', 'nunique'),
                      bypass_rate_subset=('NNHI', lambda x: (x > 1).mean()))
                 .reset_index()
    )

    group_stats = group_stats.merge(group_stats_subset, on='HousePrice', how='left').fillna({'group_size_subset':0, 'bypass_rate_subset':0})

    # Convert bypass rate to integer percentage
    group_stats['bypass_int'] = (group_stats['bypass_rate'] * 100).round().astype(int)
    group_stats['bypass_int_subset'] = (group_stats['bypass_rate_subset'] * 100).round().astype(int)

    # Aggregate by bypass_int
    bypass_by_bucket = group_stats.groupby('bypass_int')['group_size'].sum().reset_index().rename(columns={'group_size':'Total'})
    bypass_subset_by_bucket = group_stats.groupby('bypass_int')['group_size_subset'].sum().reset_index().rename(columns={'group_size_subset':'subset'})

    bypass_df = bypass_by_bucket.merge(bypass_subset_by_bucket, on='bypass_int', how='left').fillna(0)
    bypass_df = bypass_df.rename(columns={'bypass_int':'bypass'})
    bypass_df['Total'] = bypass_df['Total'].astype(int)
    bypass_df['subset'] = bypass_df['subset'].astype(int)
    bypass_df['Total_Percentage'] = bypass_df['Total'] / total_n
    bypass_df['subset_Percentage'] = bypass_df['subset'] / total_n

    # Fill missing buckets 0-100
    all_rates = pd.DataFrame({'bypass': list(range(0,101))})
    bypass_df = all_rates.merge(bypass_df, on='bypass', how='left').fillna(0)

    bypass_df.to_csv(OUTPUT_PATH / f"bypass_{subset_name}.csv", index=False, encoding='utf-8-sig')

    # -------------------------
    # 2) NNHI counts
    # -------------------------
    nnhi_counts = df.groupby('NNHI').agg(Total=('id','nunique')).reset_index()
    nnhi_counts_subset = df_subset.groupby('NNHI').agg(subset=('id','nunique')).reset_index()
    nnhi_df = nnhi_counts.merge(nnhi_counts_subset, on='NNHI', how='left').fillna(0)
    nnhi_range = range(int(df['NNHI'].min()), int(df['NNHI'].max())+1)
    nnhi_full = pd.DataFrame({'NNHI': list(nnhi_range)})
    nnhi_df = nnhi_full.merge(nnhi_df, on='NNHI', how='left').fillna(0)
    nnhi_df['Total'] = nnhi_df['Total'].astype(int)
    nnhi_df['subset'] = nnhi_df['subset'].astype(int)
    nnhi_df['Total_Percentage'] = nnhi_df['Total'] / total_n
    nnhi_df['subset_Percentage'] = nnhi_df['subset'] / total_n
    nnhi_df.to_csv(OUTPUT_PATH / f"NNHI_{subset_name}.csv", index=False, encoding='utf-8-sig')

    # -------------------------
    # 3) Extra distance counts
    # -------------------------
    dist_counts = df.groupby('extra_distance_group').agg(Total=('id','nunique')).reset_index()
    dist_counts_subset = df_subset.groupby('extra_distance_group').agg(subset=('id','nunique')).reset_index()
    dist_df = pd.DataFrame({'extra_distance_group': list(labels)}).merge(dist_counts, on='extra_distance_group', how='left').merge(dist_counts_subset, on='extra_distance_group', how='left').fillna(0)
    dist_df['Total'] = dist_df['Total'].astype(int)
    dist_df['subset'] = dist_df['subset'].astype(int)
    dist_df['Total_Percentage'] = dist_df['Total'] / total_n
    dist_df['subset_Percentage'] = dist_df['subset'] / total_n
    dist_df.to_csv(OUTPUT_PATH / f"extra_distance_{subset_name}.csv", index=False, encoding='utf-8-sig')


# Generate summary CSVs for top-tier and high-reputation hospitals
generate_summary_csv(df_top, "top_tier")
generate_summary_csv(df_high, "high_reputation")

print("All summary CSV files generated successfully!")

# ======================================================
# Plotting functions
# ======================================================
Fontsize_ax = 26
Fontsize_label = 20

def plot_percentage(output_path, xcol, ycol, measure):
    """Scatter plot with log-fit line"""
    plt.figure(figsize=(7,6))
    x = xcol
    y = ycol * 100
    plt.scatter(x, y, color='gray', alpha=0.6)

    log_x = np.log(x.replace(0, np.nan))
    mask = ~np.isnan(log_x)
    slope, intercept, r_value, p_value, std_err = linregress(log_x[mask], y[mask])
    y_fit = intercept + slope * log_x
    plt.plot(x, y_fit, color='red', alpha=0.6)

    p_text = "< 0.001" if p_value < 0.001 else f"= {p_value:.2e}"
    plt.text(0.47, 0.04, f'$R^2 = {r_value**2:.2f}, p {p_text}$', fontsize=Fontsize_label, transform=plt.gca().transAxes)

    if measure == "NNHI":
        plt.xlabel("NNHI", fontsize=Fontsize_ax)
    elif measure == "distance":
        plt.xlabel("Extra travel distance (km)", fontsize=Fontsize_ax)
    elif measure == "bypass":
        plt.ylim(10,33)
        plt.xlabel("Bypass rate (%)", fontsize=Fontsize_ax)

    plt.ylabel("Percentage (%)", fontsize=Fontsize_ax)
    plt.xticks(fontsize=Fontsize_label)
    plt.yticks(fontsize=Fontsize_label)

    plt.savefig(output_path, format="jpg", dpi=600, bbox_inches="tight")
    plt.close()

def plot_bar(output_path, xcol, ycol, ycol2, measure):
    """Bar plot comparing total vs subset"""
    plt.figure(figsize=(8,6))
    if measure == "NNHI":
        plt.bar(xcol, ycol*100, width=0.85, color='gray', alpha=0.6)
        plt.bar(xcol, ycol2*100, width=0.35, color='red', alpha=0.6)
        plt.xlabel("NNHI", fontsize=Fontsize_ax)
        plt.xlim(0, max(xcol))
    elif measure == "distance":
        plt.bar(xcol, ycol*100, width=0.4, color='gray', alpha=0.6)
        plt.bar(xcol, ycol2*100, width=0.15, color='red', alpha=0.6)
        plt.xlabel("Extra travel distance (km)", fontsize=Fontsize_ax)
        plt.xlim(0, 51)
    elif measure == "bypass":
        plt.bar(xcol, ycol*100, width=0.85, color='gray', alpha=0.6)
        plt.bar(xcol, ycol2*100, width=0.35, color='red', alpha=0.6)
        plt.xlabel("Bypass rate (%)", fontsize=Fontsize_ax)
        plt.xlim(0, 101)

    plt.ylabel("Percentage (%)", fontsize=Fontsize_ax)
    plt.xticks(fontsize=Fontsize_label)
    plt.yticks(fontsize=Fontsize_label)

    plt.savefig(output_path, format="jpg", dpi=600, bbox_inches="tight")
    plt.close()
