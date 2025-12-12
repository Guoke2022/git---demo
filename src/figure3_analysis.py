# ------------------------------------------------------------------------------
# figure3_analysis.py

# This script generates Figure 3 for hospital accessibility analysis:
# - Dual bar plot of bypass rate and NNHI by SES group
# - Lorenz curves and C-index (CI) for road distance, NNHI, and bypass
# ------------------------------------------------------------------------------


import pandas as pd
import numpy as np
from pathlib import Path
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


def compute_CI(low_mean, high_mean, low_se, high_se):
    """
    Compute absolute and relative differences between groups, along with
    95% confidence intervals using standard errors.

    Parameters
    ----------
    low_mean : float
        Mean value for the low-SES group.
    high_mean : float
        Mean value for the high-SES group.
    low_se : float
        Standard error for the low-SES group.
    high_se : float
        Standard error for the high-SES group.

    Returns
    -------
    diff : float
        Absolute difference between high and low means.
    abs_CI : tuple
        95% CI for the absolute difference.
    rel_diff : float
        Relative difference (percentage value).
    rel_CI : tuple
        95% CI for the relative difference.
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


def p_to_stars(p):
    """Convert a p-value into standard significance star notation."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''


def get_sig_pairs(df, value_col, group_col='SES_group', alpha=0.05):
    """
    Perform Tukey HSD test to obtain pairwise significance between SES groups.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the data.
    value_col : str
        Column to compare across groups.
    group_col : str
        Column specifying group membership.
    alpha : float
        Significance level for Tukey HSD.

    Returns
    -------
    list of tuples
        Each tuple contains (group1, group2, p-value, significance stars).
    """
    sub = df[[group_col, value_col]].dropna()

    mc = MultiComparison(sub[value_col], sub[group_col])
    tukey_res = mc.tukeyhsd(alpha=alpha)

    print(f"\n====== {value_col} Tukey HSD (alpha={alpha}) ======")
    print(tukey_res.summary())

    groups = list(mc.groupsunique)
    pairs = []
    idx = 0

    for i in range(len(groups) - 1):
        for j in range(i + 1, len(groups)):
            g1, g2 = groups[i], groups[j]
            p = float(tukey_res.pvalues[idx])
            pairs.append((g1, g2, p, p_to_stars(p)))
            idx += 1

    return pairs


def add_sig_bracket(ax, x1, x2, y, h, text, fontsize=14):
    """
        Draw a significance bracket on the plot between two x positions.
    """
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color='black', linewidth=1)
    ax.text((x1 + x2) / 2, y + h, text, ha='center', va='bottom', fontsize=fontsize)


def plot_dual_bar(df_plot, bypass_col, nnhi_col, out_name):
    """
    Generate a dual-indicator bar plot comparing bypass rate and NNHI
    across SES groups, including significance annotations.

    Parameters
    ----------
    df_plot : pandas.DataFrame
        Aggregated grid-level dataset containing SES group and metrics.
    bypass_col : str
        Name of the bypass rate column.
    nnhi_col : str
        Name of the NNHI column.
    out_name : str
        Output image filename.
    """

    # Font sizes
    Fontsize_ax = 18
    Fontsize_label = 16
    Fontsize_legend = 15

    # Tukey HSD for significance
    bypass_pairs = get_sig_pairs(df_plot, bypass_col, 'SES_group')
    nnhi_pairs = get_sig_pairs(df_plot, nnhi_col, 'SES_group')

    # Group summary statistics
    result_df = df_plot.groupby('SES_group').agg(
        bypass_mean=(bypass_col, 'mean'),
        bypass_se=(bypass_col, lambda x: x.sem()),
        nnhi_mean=(nnhi_col, 'mean'),
        nnhi_se=(nnhi_col, lambda x: x.sem())
    ).reset_index()

    result_df['SES_group'] = pd.Categorical(
        result_df['SES_group'], categories=['Low', 'Moderate', 'High'], ordered=True
    )
    result_df = result_df.sort_values('SES_group').reset_index(drop=True)

    # Extract values for Low vs High comparison
    low_bypass, high_bypass = result_df.loc[0, 'bypass_mean'], result_df.loc[2, 'bypass_mean']
    low_NNHI, high_NNHI = result_df.loc[0, 'nnhi_mean'], result_df.loc[2, 'nnhi_mean']
    low_bypass_se, high_bypass_se = result_df.loc[0, 'bypass_se'], result_df.loc[2, 'bypass_se']
    low_NNHI_se, high_NNHI_se = result_df.loc[0, 'nnhi_se'], result_df.loc[2, 'nnhi_se']

    # Compute differences
    abs_diff_bp, abs_CI_bp, rel_diff_bp, rel_CI_bp = compute_CI(
        low_bypass, high_bypass, low_bypass_se, high_bypass_se
    )
    abs_diff_n, abs_CI_n, rel_diff_n, rel_CI_n = compute_CI(
        low_NNHI, high_NNHI, low_NNHI_se, high_NNHI_se
    )

    print("\n====== Bypass difference (High - Low) ======")
    print(f"Absolute diff: {abs_diff_bp:.4f}, 95% CI: {abs_CI_bp}")
    print(f"Relative diff: {rel_diff_bp:.2%}, 95% CI: {rel_CI_bp}")

    print("\n====== NNHI difference (High - Low) ======")
    print(f"Absolute diff: {abs_diff_n:.4f}, 95% CI: {abs_CI_n}")
    print(f"Relative diff: {rel_diff_n:.2%}, 95% CI: {rel_CI_n}")

    # Plotting setup
    n_ses = len(result_df)
    bar_width = 0.6
    gap = 0.5

    x_bypass = np.arange(n_ses)
    x_nnhi = x_bypass + n_ses + gap

    bypass_means = result_df['bypass_mean'].values
    bypass_ses = result_df['bypass_se'].values
    nnhi_means = result_df['nnhi_mean'].values
    nnhi_ses = result_df['nnhi_se'].values
    ses_labels = result_df['SES_group'].astype(str).values

    fig, ax1 = plt.subplots(figsize=(9, 7))

    # Bypass error bars
    ax1.errorbar(x_bypass, bypass_means, yerr=bypass_ses * 1.96,
                 fmt='none', ecolor='black', capsize=5)
    ax1.set_ylabel('Bypass rate', fontsize=Fontsize_ax)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax1.tick_params(axis='y', labelsize=Fontsize_label)
    ax1.set_ylim(0.5, 1.0)
    ax1.set_yticks(np.arange(0.5, 1.01, 0.1))

    # NNHI on secondary axis
    ax2 = ax1.twinx()
    ax2.errorbar(x_nnhi, nnhi_means, yerr=nnhi_ses * 1.96,
                 fmt='none', ecolor='black', capsize=5)
    ax2.set_ylabel('NNHI', fontsize=Fontsize_ax)
    ax2.tick_params(axis='y', labelsize=Fontsize_label)
    ax2.set_ylim(15, 22)
    ax2.set_yticks(np.arange(15, 23, 1))

    # X-axis labels centered
    ax1.set_xticks([x_bypass.mean(), x_nnhi.mean()])
    ax1.set_xticklabels(['Bypass rate', 'NNHI'], fontsize=Fontsize_ax)

    # Legend
    legend_handles = [
        Line2D([0], [0], marker='s', linestyle='None', markersize=10,
               markerfacecolor='#EDE29B97', markeredgecolor='black', label='Bypass rate'),
        Line2D([0], [0], marker='s', linestyle='None', markersize=10,
               markerfacecolor='#3878C197', markeredgecolor='black', label='NNHI')]

    fig.legend(handles=legend_handles, bbox_to_anchor=(0.90, 0.97),
               loc='upper right', fontsize=Fontsize_legend, frameon=False, ncol=2)

    # Text labels above bars
    pad_b1 = 0.04 * (ax1.get_ylim()[1] - ax1.get_ylim()[0])
    pad_n1 = 0.04 * (ax2.get_ylim()[1] - ax2.get_ylim()[0])

    display_labels = ['Mid' if s == 'Moderate' else s for s in ses_labels]

    for x, m, se, lab in zip(x_bypass, bypass_means, bypass_ses, display_labels):
        ax1.text(x, m + se * 1.96 + pad_b1, lab, ha='center', fontsize=Fontsize_label)

    for x, m, se, lab in zip(x_nnhi, nnhi_means, nnhi_ses, display_labels):
        ax2.text(x, m + se * 1.96 + pad_n1, lab, ha='center', fontsize=Fontsize_label)

    # Significance brackets
    xmap_bp = {g: x for g, x in zip(ses_labels, x_bypass)}
    xmap_nn = {g: x for g, x in zip(ses_labels, x_nnhi)}

    tops_bp = bypass_means + bypass_ses * 1.96 + pad_b1
    tops_nn = nnhi_means + nnhi_ses * 1.96 + pad_n1

    base_bp = tops_bp.max() + pad_b1
    base_nn = tops_nn.max() + pad_n1

    sig_bp = [(g1, g2, s) for g1, g2, p, s in bypass_pairs if s]
    sig_nn = [(g1, g2, s) for g1, g2, p, s in nnhi_pairs if s]

    sig_bp.sort(key=lambda t: abs(xmap_bp[t[0]] - xmap_bp[t[1]]))
    sig_nn.sort(key=lambda t: abs(xmap_nn[t[0]] - xmap_nn[t[1]]))

    pad_br_bp = 0.04 * (ax1.get_ylim()[1] - ax1.get_ylim()[0])
    pad_br_nn = 0.04 * (ax2.get_ylim()[1] - ax2.get_ylim()[0])

    for level, (g1, g2, s) in enumerate(sig_bp):
        y = base_bp + level * pad_br_bp
        add_sig_bracket(ax1, xmap_bp[g1], xmap_bp[g2], y, pad_br_bp * 0.4, s, fontsize=Fontsize_label)

    for level, (g1, g2, s) in enumerate(sig_nn):
        y = base_nn + level * pad_br_nn
        add_sig_bracket(ax2, xmap_nn[g1], xmap_nn[g2], y, pad_br_nn * 0.4, s, fontsize=Fontsize_label)

    # X limits
    ax1.set_xlim(x_bypass[0] - bar_width, x_nnhi[-1] + bar_width)

    plt.tight_layout()
    plt.savefig(out_name, format='jpg', dpi=600, bbox_inches='tight', pad_inches=0.1)
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

# =================== Main Execution ===================

if __name__ == "__main__":

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