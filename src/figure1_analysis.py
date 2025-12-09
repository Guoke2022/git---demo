"""
plot_figure1.py

This script generates key figures for the paper.
It includes data preprocessing, calculation of extra travel ratios, 
and visualization of travel distance, extra distance, and bypass rate
for cities and city groups.

"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from pathlib2 import Path
import seaborn as sns
from pypinyin import lazy_pinyin

pd.set_option('display.max_columns', None)

# ========== Plotting settings ==========
plt.rcParams['font.family'] = 'Arial'
FONT_SIZE_AX = 22
FONT_SIZE_LEGEND = 20
FONT_SIZE_LABEL = 18
FONT_SIZE_CITY = 22


# ========== User paths ==========
DATA_PATH = Path("YOUR_DATA_PATH_HERE")  # Replace with your CSV folder/file path
OUTPUT_PATH = Path("YOUR_OUTPUT_PATH_HERE")  # Replace with output folder
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# ========== City mappings ==========
city_order = ['Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen',
              'Wuhan', 'Chengdu',
              'Qiqihar', 'Pu’er',
              'Haidong', 'Turpan', 'Shigatse'][::-1]

city_map = {
    '北京市': 'Beijing',
    '上海市': 'Shanghai',
    '广州市': 'Guangzhou',
    '深圳市': 'Shenzhen',
    '武汉市': 'Wuhan',
    '成都市': 'Chengdu',
    '齐齐哈尔市': 'Qiqihar',
    '普洱市': 'Pu’er',
    '海东市': 'Haidong',
    '吐鲁番市': 'Turpan',
    '日喀则市': 'Shigatse'
}

city_group_map = {
    'Beijing': 'Mega city',
    'Shanghai': 'Mega city',
    'Shenzhen': 'Mega city',
    'Guangzhou': 'Mega city',
    'Chengdu': 'Major city',
    'Wuhan': 'Major city',
    'Qiqihar': 'Medium city',
    'Pu’er': 'Medium city',
    'Haidong': 'Small city',
    'Shigatse': 'Small city',
    'Turpan': 'Small city'
}

group_order = ['Mega city', 'Major city', 'Medium city', 'Small city'][::-1]


# ========== Functions ==========
def calc_extra_ratio(data, n_threshold, city_order, bypass_rate_df):
    """
    Calculate extra travel ratio per city.
    """
    subset = data[data['NNHI'] > n_threshold].copy()
    if subset.empty:
        cols = ['City', 'Bypass rate', 'Travel distance (km)', 'Closest distance (km)', 'Extra travel distance']
        return pd.DataFrame(columns=cols)

    subset['Extra_Distance'] = subset['home_road_dist'] - subset['road_closest']

    result = (
        subset.groupby('City_Eng')
        .agg(
            mean_home_road_dist=('home_road_dist', 'mean'),
            mean_road_closest=('road_closest', 'mean')
        )
        .reset_index()
    )
    result['Extra_ratio'] = (result['mean_home_road_dist'] / result['mean_road_closest'] - 1)

    result = result.merge(bypass_rate_df, on='City_Eng', how='left')
    result['City_Eng'] = pd.Categorical(result['City_Eng'], categories=city_order, ordered=True)
    result = result.sort_values('City_Eng').reset_index(drop=True)

    result['Bypass rate'] = (result['Bypass rate'] * 100).round(2).astype(str) + '%'
    result['mean_home_road_dist'] = result['mean_home_road_dist'].round(2)
    result['mean_road_closest'] = result['mean_road_closest'].round(2)
    result['Extra_ratio'] = (result['Extra_ratio'] * 100).round(2).astype(str) + '%'

    result = result.rename(columns={
        'City_Eng': 'City',
        'mean_home_road_dist': 'Travel distance (km)',
        'mean_road_closest': 'Closest distance (km)',
        'Extra_ratio': 'Extra travel distance'
    })
    return result[['City', 'Bypass rate', 'Travel distance (km)', 'Closest distance (km)', 'Extra travel distance']]


def calc_extra_ratio_group(data, n_threshold, city_group_map, group_order, bypass_rate_df):
    """
    Calculate extra travel ratio per city group.
    """
    data = data.copy()
    data['City_Group'] = data['City_Eng'].map(city_group_map)

    subset = data[data['NNHI'] > n_threshold].copy()
    if subset.empty:
        cols = ['City Group', 'Bypass rate', 'Travel distance (km)', 'Closest distance (km)', 'Extra travel distance']
        return pd.DataFrame(columns=cols)

    subset['Extra_Distance'] = subset['home_road_dist'] - subset['road_closest']

    result = (
        subset.groupby('City_Group')
        .agg(
            mean_home_road_dist=('home_road_dist', 'mean'),
            mean_road_closest=('road_closest', 'mean')
        )
        .reset_index()
    )
    result['Extra_ratio'] = (result['mean_home_road_dist'] / result['mean_road_closest'] - 1)

    bypass_rate_df = bypass_rate_df.copy()
    bypass_rate_df['City_Group'] = bypass_rate_df['City_Eng'].map(city_group_map)
    bypass_rate_group = (
        bypass_rate_df.groupby('City_Group')['Bypass rate'].mean().reset_index()
    )
    result = result.merge(bypass_rate_group, on='City_Group', how='left')

    result['City_Group'] = pd.Categorical(result['City_Group'], categories=group_order, ordered=True)
    result = result.sort_values('City_Group').reset_index(drop=True)

    result['Bypass rate'] = (result['Bypass rate'] * 100).round(2).astype(str) + '%'
    result['mean_home_road_dist'] = result['mean_home_road_dist'].round(2)
    result['mean_road_closest'] = result['mean_road_closest'].round(2)
    result['Extra_ratio'] = (result['Extra_ratio'] * 100).round(2).astype(str) + '%'

    result = result.rename(columns={
        'City_Group': 'City Group',
        'mean_home_road_dist': 'Travel distance (km)',
        'mean_road_closest': 'Closest distance (km)',
        'Extra_ratio': 'Extra travel distance'
    })
    return result[['City Group', 'Bypass rate', 'Travel distance (km)', 'Closest distance (km)', 'Extra travel distance']]


def plot_three_panel(df_plot, y_col, save_path):
    """
    Plot three-panel figure: Bypass rate, Extra travel distance, and Travel distance.
    """
    if df_plot.empty:
        print(f"⚠️ Input table is empty, skipped: {save_path.name}")
        return

    df_plot = df_plot.copy()

    # Convert percentage strings to float
    def to_float_percent(s):
        if pd.isna(s):
            return np.nan
        if isinstance(s, str) and s.strip().endswith('%'):
            return float(s.strip().replace('%', ''))
        return float(s)

    df_plot['Extra travel distance'] = df_plot['Extra travel distance'].apply(to_float_percent).astype(float)
    df_plot['Bypass rate'] = df_plot['Bypass rate'].apply(to_float_percent).astype(float)

    # Set categorical order
    if y_col == 'City':
        df_plot[y_col] = pd.Categorical(df_plot[y_col], categories=city_order, ordered=True)
    else:
        df_plot[y_col] = pd.Categorical(df_plot[y_col], categories=group_order, ordered=True)
    df_plot = df_plot.sort_values(y_col).reset_index(drop=True)

    # Create figure
    if y_col == 'City':
        fig = plt.figure(figsize=(17, 5.8))
    else:
        fig = plt.figure(figsize=(17, 2.4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1.8, 1], wspace=0.06)

    # Middle panel: Travel distances
    ax1 = plt.subplot(gs[1])
    for i in range(len(df_plot)):
        ax1.hlines(y=i, xmin=df_plot['Closest distance (km)'].iat[i],
                   xmax=df_plot['Travel distance (km)'].iat[i],
                   color='gray', linestyle='-', linewidth=8, alpha=0.5)
        if i == 0:
            ax1.plot(df_plot['Closest distance (km)'].iat[i], i, 'o', color='#426885', markersize=8, label='The nearest hospital')
            ax1.plot(df_plot['Travel distance (km)'].iat[i], i, 'o', color='#bf2f24', markersize=8, label='Actually visited')

    ax1.set_xlabel('Travel distance (km)', fontsize=FONT_SIZE_AX)
    ax1.set_yticks(range(len(df_plot)))
    ax1.set_yticklabels([])
    ax1.tick_params(axis='x', labelsize=FONT_SIZE_LABEL)
    ax1.tick_params(axis='y', length=0)

    # Right panel: Extra travel distance
    ax2 = plt.subplot(gs[2])
    ax2.hlines(y=df_plot[y_col], xmin=0, xmax=(df_plot['Extra travel distance']/100),
               color='grey', linestyle='dashed', linewidth=2, alpha=0.5, zorder=5)
    ax2.scatter(df_plot['Extra travel distance']/100, df_plot[y_col],
                s=80, marker='o', facecolors='none', edgecolors='#bf2f24',
                linewidth=2.2, zorder=20, label='Extra travel distance')

    ax2.invert_xaxis()
    ax2.set_xlabel('Extra travel distance (%)', fontsize=FONT_SIZE_AX)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_ticks_position('right')
    ax2.set_yticklabels([])
    ax2.tick_params(axis='x', labelsize=FONT_SIZE_LABEL)
    ax2.tick_params(axis='y', length=0)
    ax2.set_xlim(left=0, right=df_plot['Extra travel distance'].max()/100 + 0.2)
    ax2.xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))

    # Left panel: Bypass rate
    ax3 = plt.subplot(gs[0])
    ax3.hlines(y=df_plot[y_col], xmin=0, xmax=(df_plot['Bypass rate']/100),
               color='grey', linestyle='dashed', linewidth=2, alpha=0.5, zorder=5)
    ax3.scatter(df_plot['Bypass rate']/100, df_plot[y_col],
                s=80, marker='o', facecolors='none', edgecolors='#426885',
                linewidth=2.2, zorder=20, label='Bypass rate')

    ax3.invert_xaxis()
    ax3.set_xlabel('Bypass rate', fontsize=FONT_SIZE_AX)
    ax3.yaxis.set_ticks_position('left')
    ax3.set_yticks(range(len(df_plot[y_col])))
    ax3.set_yticklabels(df_plot[y_col], fontsize=FONT_SIZE_CITY)
    ax3.tick_params(axis='x', labelsize=FONT_SIZE_LABEL)
    ax3.set_xlim(left=0, right=1)
    ax3.xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))

    plt.tight_layout()
    plt.savefig(save_path, format='jpg', dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"✅ Saved: {save_path.name}")


# ========== Main script ==========
def main():
    # Load data
    df = pd.read_csv(DATA_PATH, encoding='utf-8')

    # Check required columns
    required_cols = ['city', 'id', 'NNHI', 'home_road_dist', 'road_closest']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df['City_Eng'] = df['city'].map(city_map)
    df.loc[df['road_closest'] < 0.05, 'road_closest'] = 0.05

    # Bypass rate per city
    bypass_rate = df.groupby('City_Eng').apply(lambda x: (x['NNHI'] > 1).mean()).reset_index(name='Bypass rate')

    # Calculate extra ratios
    extra_city = calc_extra_ratio(df, 1, city_order, bypass_rate)
    extra_group = calc_extra_ratio_group(df, 1, city_group_map, group_order, bypass_rate)

    # Save CSVs
    extra_city.to_csv(OUTPUT_PATH / "Extra_ratio_city.csv", index=False, encoding='utf-8-sig')
    extra_group.to_csv(OUTPUT_PATH / "Extra_ratio_group.csv", index=False, encoding='utf-8-sig')
    print("✅ CSV outputs saved.")

    # Plot
    if not extra_city.empty:
        plot_three_panel(extra_city, 'City', OUTPUT_PATH / "Extra_ratio_city.jpg")
    else:
        print("City-level results empty, skipped plotting.")

    if not extra_group.empty:
        plot_three_panel(extra_group, 'City Group', OUTPUT_PATH / "Extra_ratio_group.jpg")
    else:
        print("Group-level results empty, skipped plotting.")

    print("✅ All processing completed!")


if __name__ == "__main__":
    main()