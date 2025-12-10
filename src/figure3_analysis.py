# ------------------------------------------------------------------------------
# figure3_analysis.py

# This script generates Figure 3 for hospital accessibility analysis:
# - Dual bar plot of bypass rate and NNHI by SES group
# - Lorenz curves and C-index (CI) for road distance, NNHI, and bypass
# ------------------------------------------------------------------------------


import pandas as pd
import numpy as np
from pathlib2 import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.integrate import simpson

pd.set_option('display.max_columns', None)

# --------------------------
# Plot settings
# --------------------------
plt.rcParams['font.family'] = 'Arial'
Fontsize_ax = 28
Fontsize_legend = 26
Fontsize_label = 20
LineWidth = 2
custom_colors = ['#d92523', '#70c17f', '#689EA2', '#f7ae55', '#2e7ebb', '#7262ac', '#5f5f5f']


# --------------------------
# Paths
# --------------------------
# Folder to save output figures
PATH_RESULT = Path("./output/Figure3/")
PATH_RESULT.mkdir(parents=True, exist_ok=True)

# Input data files
DF_PATH = Path("./data/NNHI_summary.csv")          # Replace with your NNHI summary CSV
SES_TABLE = Path("./data/SES_grid_summary.csv")    # Replace with your SES grid CSV


# --------------------------
# Functions
# --------------------------
def compute_CI(low_mean, high_mean, low_se, high_se):
    """
    Compute absolute and relative difference with 95% CI
    """
    diff = high_mean - low_mean
    se_diff = np.sqrt(low_se**2 + high_se**2)
    ci_low = diff - 1.96 * se_diff
    ci_high = diff + 1.96 * se_diff
    rel_diff = diff / low_mean
    se_rel = se_diff / low_mean
    rel_ci_low = rel_diff - 1.96 * se_rel
    rel_ci_high = rel_diff + 1.96 * se_rel
    return diff, (ci_low, ci_high), rel_diff, (rel_ci_low, rel_ci_high)


def trans_pinyin(city_name):
    """
    Convert Chinese city name to pinyin
    """
    from pypinyin import lazy_pinyin
    if city_name == '齐齐哈尔市':
        return "Qiqihar"
    else:
        return ''.join(lazy_pinyin(city_name)).capitalize()[:-3]


def aggregate_by_houseprice(df_in):
    """
    Aggregate data by HousePrice
    """
    def calc_metrics(group):
        total = len(group)
        if total == 0:
            return pd.Series({'bypass': np.nan, 'NNHI': np.nan, 'home_road_dist': np.nan})
        bypass = (group['NNHI'] > 1).mean()
        NNHI = group['NNHI'].mean()
        home_road_dist = group['home_road_dist'].mean()
        return pd.Series({'bypass': bypass, 'NNHI': NNHI, 'home_road_dist': home_road_dist})
    res = df_in.groupby('HousePrice').apply(calc_metrics).reset_index()
    return res


def plot_dual_bar(df_plot, bypass_col, nnhi_col, out_file):
    """
    Dual bar plot of bypass rate and NNHI by SES group with 95% CI
    """
    # Compute mean and SE
    result_df = df_plot.groupby('SES_group').agg(
        bypass_mean=(bypass_col, 'mean'),
        bypass_se=(bypass_col, lambda x: x.sem()),
        nnhi_mean=(nnhi_col, 'mean'),
        nnhi_se=(nnhi_col, lambda x: x.sem())
    ).reset_index()
    result_df['SES_group'] = pd.Categorical(result_df['SES_group'], categories=['Low', 'Moderate', 'High'], ordered=True)
    result_df = result_df.sort_values('SES_group').reset_index(drop=True)

    # Compute differences between High and Low
    low_bypass = result_df.loc[result_df['SES_group'] == 'Low', 'bypass_mean'].values[0]
    high_bypass = result_df.loc[result_df['SES_group'] == 'High', 'bypass_mean'].values[0]
    low_bypass_se = result_df.loc[result_df['SES_group'] == 'Low', 'bypass_se'].values[0]
    high_bypass_se = result_df.loc[result_df['SES_group'] == 'High', 'bypass_se'].values[0]
    compute_CI(low_bypass, high_bypass, low_bypass_se, high_bypass_se)

    low_nnhi = result_df.loc[result_df['SES_group'] == 'Low', 'nnhi_mean'].values[0]
    high_nnhi = result_df.loc[result_df['SES_group'] == 'High', 'nnhi_mean'].values[0]
    low_nnhi_se = result_df.loc[result_df['SES_group'] == 'Low', 'nnhi_se'].values[0]
    high_nnhi_se = result_df.loc[result_df['SES_group'] == 'High', 'nnhi_se'].values[0]
    compute_CI(low_nnhi, high_nnhi, low_nnhi_se, high_nnhi_se)

    # Plot
    bar_width = 0.35
    x = np.arange(len(result_df))
    fig, ax1 = plt.subplots(figsize=(9, 7))

    # Left y-axis: Bypass rate
    ax1.bar(x - bar_width/2, result_df['bypass_mean'], width=bar_width, color='#EDE29B97', label='Bypass rate')
    ax1.errorbar(x - bar_width/2, result_df['bypass_mean'], yerr=result_df['bypass_se']*1.96, fmt='none', ecolor='black', capsize=5)
    ax1.set_ylabel('Bypass rate', fontsize=Fontsize_ax)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax1.set_ylim(0.5, 0.9)
    ax1.set_yticks(np.arange(0.5, 0.91, 0.1))
    ax1.tick_params(axis='y', labelsize=Fontsize_label)

    # Right y-axis: NNHI
    ax2 = ax1.twinx()
    ax2.bar(x + bar_width/2, result_df['nnhi_mean'], width=bar_width, color='#3878C197', label='NNHI')
    ax2.errorbar(x + bar_width/2, result_df['nnhi_mean'], yerr=result_df['nnhi_se']*1.96, fmt='none', ecolor='black', capsize=5)
    ax2.set_ylabel('NNHI', fontsize=Fontsize_ax)
    ax2.set_ylim(15, 21)
    ax2.set_yticks(np.arange(15, 22, 1))
    ax2.tick_params(axis='y', labelsize=Fontsize_label)

    # X-axis
    ax1.set_xticks(x)
    ax1.set_xticklabels(result_df['SES_group'], fontsize=Fontsize_ax)
    ax1.set_xlabel('SES group', fontsize=Fontsize_ax)

    # Legend
    fig.legend(bbox_to_anchor=(0.90, 0.97), loc='upper right', fontsize=Fontsize_legend, frameon=False)
    plt.tight_layout()
    plt.savefig(out_file, format='jpg', dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def plot_lorenz_curve(df, value_col, city_list, colors, ylabel, out_file):
    """
    Plot Lorenz curve and compute CI for multiple cities
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    city_pinyins = []
    area_coefficients = []

    for idx, CITY in enumerate(city_list):
        df_city = df[df['city'] == CITY].copy()
        if df_city.empty:
            continue
        city_pinyin = trans_pinyin(CITY)
        city_pinyins.append(city_pinyin)
        df_sorted = df_city.sort_values('HousePrice')
        cumulative_rank = np.cumsum(df_sorted['Weight']) / np.sum(df_sorted['Weight'])
        cumulative_value = np.cumsum(df_sorted[value_col]) / np.sum(df_sorted[value_col])
        ax.plot(cumulative_rank, cumulative_value, color=colors[idx], label=city_pinyin, linewidth=LineWidth)
        area_curve = simpson(cumulative_value, cumulative_rank)
        area_coefficients.append((0.5 - area_curve)/0.5)

    ax.plot([0,1],[0,1], linestyle='--', color='gray', linewidth=LineWidth)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xlabel('Cumulative population, ranked by house price', fontsize=Fontsize_ax)
    ax.set_ylabel(ylabel, fontsize=Fontsize_ax)
    plt.xticks(fontsize=Fontsize_label)
    plt.yticks(fontsize=Fontsize_label)
    legend_labels = [f'CI:{area_coefficients[i]:.2f}' for i in range(len(city_pinyins))]
    ax.legend(labels=legend_labels, loc='lower right', fontsize=Fontsize_label, frameon=False)

    # Save results
    pd.DataFrame({'City': city_pinyins, 'CI': area_coefficients}).to_csv(out_file.with_suffix('.csv'), index=False, float_format='%.2f')
    plt.savefig(out_file, format='jpg', dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()


# --------------------------
# Load data
# --------------------------
df = pd.read_csv(DF_PATH, encoding='utf-8')
ses_data = pd.read_csv(SES_TABLE, encoding='utf-8')[['HousePrice', 'HousePrice_262cut_grid']]

grid_agg = aggregate_by_houseprice(df)
grid_agg = pd.merge(grid_agg, ses_data, on='HousePrice', how='left').drop_duplicates(subset=['HousePrice'])
ses_map = {1.0: 'High', 0.8: 'Moderate', 0.2: 'Low'}
grid_agg['SES_group'] = grid_agg['HousePrice_262cut_grid'].map(ses_map)
grid_agg = grid_agg[grid_agg['SES_group'].isin(['Low', 'Moderate', 'High'])].copy()

# --------------------------
# Dual bar plot (bypass & NNHI)
# --------------------------
plot_dual_bar(grid_agg, 'bypass', 'NNHI', PATH_RESULT/'3.1_bypass_NNHI.jpg')

# --------------------------
# Prepare for Lorenz curves
# --------------------------
city_list = ['北京市', '上海市', '深圳市', '广州市', '成都市', '武汉市', '齐齐哈尔市']
df_all = df[['id', 'name', 'NNHI', 'road_closest', 'HousePrice', 'city']].copy()
df_all['Weight'] = 1

grouped = df_all.groupby("HousePrice").agg(bypass=("NNHI", lambda x: (x>1).mean())).reset_index()
df_all = df_all.merge(grouped, on="HousePrice", how="left")

# --------------------------
# Lorenz curves
# --------------------------
plot_lorenz_curve(df_all, 'road_closest', city_list, custom_colors, 'Cumulative hospital accessibility', PATH_RESULT/'3.6_road_distance_Lorenz.jpg')
plot_lorenz_curve(df_all, 'NNHI', city_list, custom_colors, 'Cumulative NNHI', PATH_RESULT/'3.7_NNHI_Lorenz.jpg')
plot_lorenz_curve(df_all, 'bypass', city_list, custom_colors, 'Cumulative bypass', PATH_RESULT/'3.8_bypass_Lorenz.jpg')