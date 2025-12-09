"""
Sensitivity Analysis for Hospital Bypass Behavior and SES Disparities
---------------------------------------------------------------------

This script performs two major analyses:
1. Sensitivity analysis of hospital bypass behavior using alternative bypass metrics.
2. Sensitivity analysis of how patient identification thresholds influence bypass prevalence
   and socioeconomic disparities.

Main functionalities include:
- Computing bypass prevalence under different NNHI thresholds.
- Computing extra travel distance ratios at both city-level and city-group-level.
- Evaluating robustness of bypass metrics under alternative filtering conditions.

All outputs are saved as CSV tables for further statistical analysis and visualization.

"""

import pandas as pd
from pathlib import Path


# ================================================================
# Configuration
# ================================================================

# Font settings
FONTSIZE_AX = 20
FONTSIZE_LEGEND = 20
FONTSIZE_CITY = 22
FONTSIZE_LABEL = 18

# Input data (replace these paths when uploading to GitHub)
PATH_TOTAL_TABLE = Path("path/to/total_table.csv")
PATH_MAIN_TABLE = Path("path/to/main_table_no_companion.csv")
PATH_ID_LIST_DIR = Path("path/to/id_list_directory/")
PATH_OUTPUT = Path("path/to/output/")
PATH_OUTPUT.mkdir(parents=True, exist_ok=True)

# City name mapping
CITY_ORDER = [
    'Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen',
    'Wuhan', 'Chengdu', 'Qiqihar', 'Pu’er',
    'Haidong', 'Turpan', 'Shigatse'
][::-1]

CITY_MAP = {
    '北京市': 'Beijing', '上海市': 'Shanghai', '广州市': 'Guangzhou', '深圳市': 'Shenzhen',
    '武汉市': 'Wuhan', '成都市': 'Chengdu', '齐齐哈尔市': 'Qiqihar', '普洱市': 'Pu’er',
    '海东市': 'Haidong', '吐鲁番市': 'Turpan', '日喀则市': 'Shigatse'
}

CITY_GROUP_MAP = {
    'Beijing': 'Mega city', 'Shanghai': 'Mega city',
    'Shenzhen': 'Mega city', 'Guangzhou': 'Mega city',
    'Chengdu': 'Major city', 'Wuhan': 'Major city',
    'Qiqihar': 'Medium city', 'Pu’er': 'Medium city',
    'Haidong': 'Small city', 'Shigatse': 'Small city', 'Turpan': 'Small city'
}

GROUP_ORDER = ['Mega city', 'Major city', 'Medium city', 'Small city']


# ================================================================
# Utility Functions
# ================================================================

def compute_bypass_rate(data: pd.DataFrame, threshold: float):
    """Compute bypass rate based on NNHI threshold."""
    return (
        data.groupby('City_Eng')
        .apply(lambda x: (x['NNHI'] > threshold).mean())
        .reset_index(name='Bypass rate')
    )


def compute_extra_ratio_city(data, n_threshold, dist_threshold, city_order, bypass_rate_df):
    """
    Compute extra travel ratio for each city.
    NNHI > n_threshold and extra travel distance > dist_threshold.
    """
    subset = data[data['NNHI'] > n_threshold].copy()
    subset['Extra_Distance'] = subset['home_road_dist'] - subset['road_closest']
    subset = subset[subset['Extra_Distance'] >= dist_threshold]

    result = (
        subset.groupby('City_Eng')
        .agg(
            mean_home_road_dist=('home_road_dist', 'mean'),
            mean_road_closest=('road_closest', 'mean')
        )
        .reset_index()
    )

    result['Extra_ratio'] = (result['mean_home_road_dist'] /
                             result['mean_road_closest'] - 1)

    # Merge bypass rate
    result = result.merge(bypass_rate_df, on='City_Eng', how='left')

    # Sort by predefined city order
    result['City_Eng'] = pd.Categorical(
        result['City_Eng'], categories=city_order, ordered=True
    )
    result = result.sort_values('City_Eng')

    # Format columns
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

    return result


def compute_extra_ratio_group(data, n_threshold, dist_threshold,
                              city_group_map, group_order, bypass_rate_df):
    """Compute extra travel ratio aggregated at city-group level."""
    data = data.copy()
    data['City_Group'] = data['City_Eng'].map(city_group_map)

    subset = data[data['NNHI'] > n_threshold].copy()
    subset['Extra_Distance'] = subset['home_road_dist'] - subset['road_closest']
    subset = subset[subset['Extra_Distance'] >= dist_threshold]

    result = (
        subset.groupby('City_Group')
        .agg(
            mean_home_road_dist=('home_road_dist', 'mean'),
            mean_road_closest=('road_closest', 'mean')
        )
        .reset_index()
    )

    result['Extra_ratio'] = (result['mean_home_road_dist'] /
                             result['mean_road_closest'] - 1)

    # Compute group-level bypass rate
    df_br = bypass_rate_df.copy()
    df_br['City_Group'] = df_br['City_Eng'].map(city_group_map)
    bypass_rate_group = (
        df_br.groupby('City_Group')['Bypass rate'].mean().reset_index()
    )

    result = result.merge(bypass_rate_group, on='City_Group', how='left')

    # Sort
    result['City_Group'] = pd.Categorical(
        result['City_Group'], categories=group_order, ordered=True
    )
    result = result.sort_values('City_Group')

    # Format
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

    return result


# ================================================================
# Load data for Analysis 1
# ================================================================

data_total = pd.read_csv(PATH_TOTAL_TABLE)
data_total['City_Eng'] = data_total['城市'].map(CITY_MAP)

# Avoid zero-distances
data_total.loc[data_total['road_closest'] < 0.05, 'road_closest'] = 0.05


# ---- Bypass rate tables ----
bypass_rate_N1 = compute_bypass_rate(data_total, threshold=1)
bypass_rate_N3 = compute_bypass_rate(data_total, threshold=3)


# ---- City-level outputs ----
outputs_city = {
    'Extra_ratio_N_1_D_5.csv': compute_extra_ratio_city(
        data_total, 1, 5, CITY_ORDER, bypass_rate_N1
    ),
    'Extra_ratio_N_1_D_3.csv': compute_extra_ratio_city(
        data_total, 1, 3, CITY_ORDER, bypass_rate_N1
    ),
    'Extra_ratio_N_3_D_5.csv': compute_extra_ratio_city(
        data_total, 3, 5, CITY_ORDER, bypass_rate_N3
    ),
    'Extra_ratio_N_3_D_3.csv': compute_extra_ratio_city(
        data_total, 3, 3, CITY_ORDER, bypass_rate_N3
    ),
}

for fn, df_out in outputs_city.items():
    df_out.to_csv(PATH_OUTPUT / fn, index=False, encoding='utf-8-sig')
    print(f"Saved: {fn}")


# ---- City-group outputs ----
outputs_group = {
    'Extra_ratio_Group_N_1_D_5.csv': compute_extra_ratio_group(
        data_total, 1, 5, CITY_GROUP_MAP, GROUP_ORDER, bypass_rate_N1
    ),
    'Extra_ratio_Group_N_1_D_3.csv': compute_extra_ratio_group(
        data_total, 1, 3, CITY_GROUP_MAP, GROUP_ORDER, bypass_rate_N1
    ),
    'Extra_ratio_Group_N_3_D_5.csv': compute_extra_ratio_group(
        data_total, 3, 5, CITY_GROUP_MAP, GROUP_ORDER, bypass_rate_N3
    ),
    'Extra_ratio_Group_N_3_D_3.csv': compute_extra_ratio_group(
        data_total, 3, 3, CITY_GROUP_MAP, GROUP_ORDER, bypass_rate_N3
    ),
}

for fn, df_out in outputs_group.items():
    df_out.to_csv(PATH_OUTPUT / fn, index=False, encoding='utf-8-sig')
    print(f"Saved: {fn}")


# ================================================================
# Load data for Analysis 2
# ================================================================

df_main = pd.read_csv(PATH_MAIN_TABLE)
df_main['City_Eng'] = df_main['city'].map(CITY_MAP)
df_main.loc[df_main['road_closest'] < 0.05, 'road_closest'] = 0.05


# ================================================================
# End of file
# ================================================================
