# ------------------------------------------------------------------------------
# figure4_analysis.py
#
# Purpose:
#   - Compute experienced segregation (ES) and income index (as ESC in the manuscript)
#   - Bin HousePrice into quantiles for grouping
#   - Generate Figure 4: boxplots of ES and II(income index) vs Bypass and Hospital accessibility
# ------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.patches as mpatches
pd.set_option('display.max_columns', None)

# -------------------------
# Helper functions
# -------------------------
def scale_to_0_1(data):
    """Scale a series to 0-1"""
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def scale_to_0_10(data):
    """Scale a series to 0-10"""
    return scale_to_0_1(data) * 10

# -------------------------
# Paths
# -------------------------
DATA_PATH = Path("./data")                 # input CSV folder
OUTPUT_PATH = Path("./output/Figure4")    # output folder
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

ACCOMPANY_CSV = DATA_PATH / "accompany_list.csv"  # IDs to exclude
CITY_LIST = ['Qiqihar', 'Chengdu', 'Wuhan', 'Shenzhen', 'Shanghai', 'Guangzhou', 'Beijing']

# -------------------------
# Read accompany IDs
# -------------------------
df_accompany = pd.read_csv(ACCOMPANY_CSV, encoding='utf-8')
accompany_ids = set(df_accompany['id'])

# -------------------------
# Process each city
# -------------------------
PATH_RESULT3 = DATA_PATH / "ES_data"  # store city-level processed CSVs
PATH_RESULT3.mkdir(parents=True, exist_ok=True)
dfs = []

for CITY in CITY_LIST:
    CITY_PATH = DATA_PATH / CITY / "nearest_hospitals"  # input folder for each city
    data_file = CITY_PATH / f"{CITY}_road_NNHI.csv"
    data = pd.read_csv(data_file, encoding='utf-8')

    df = data[['id', 'name', 'HousePrice', 'home_road_dist', 'NNHI', 'road_closest', 'road_second', 'road_third']]
    df = df[~df['id'].isin(accompany_ids)]  # remove accompany IDs

    # Bin HousePrice into 10 quantiles (0.1~1)
    df['HousePrice_10cut'] = pd.qcut(df['HousePrice'], 10, labels=np.arange(0.1, 1.1, 0.1)).astype(float)
    # Bin into 3 quantiles
    df['HousePrice_3cut'] = pd.qcut(df['HousePrice'], 3, labels=np.round([0.33, 0.66, 1.0], 2)).astype(float)

    # Compute bypass rate for each HousePrice
    df['bypass'] = df.groupby('HousePrice')['NNHI'].transform(lambda x: (x > 1).mean())

    # Compute experienced segregation index ES
    EI_list = []
    for name, df_group in df.groupby('name'):
        n = len(df_group)
        EI_values = [np.sum(np.abs(row - df_group['HousePrice_10cut'].values)) / (n-1)
                     for row in df_group['HousePrice_10cut']]
        EI_list.extend(zip(df_group['id'], EI_values))
    EI_df = pd.DataFrame(EI_list, columns=['id', 'EI'])
    df = df.merge(EI_df.groupby('id')['EI'].mean().reset_index(), on='id', how='left')
    df['ES'] = 1 - df['EI']

    # Compute hospital average income index II
    II_list = []
    for name, df_group in df.groupby('name'):
        II = df_group['HousePrice_10cut'].mean()
        II_list.extend(zip(df_group['id'], [II]*len(df_group)))
    II_df = pd.DataFrame(II_list, columns=['id', 'II'])
    df = df.merge(II_df.groupby('id')['II'].mean().reset_index(), on='id', how='left')

    # Assign 3-bin HousePrice cut for 20%-80% groups
    df = df.drop_duplicates(subset=['id', 'name'])
    df = df.sort_values(by='HousePrice')
    n_total = len(df)
    boundary1 = df.iloc[int(n_total*0.2)]['HousePrice']
    boundary2 = df.iloc[int(n_total*0.8)]['HousePrice']
    df['HousePrice_262cut'] = pd.cut(df['HousePrice'], bins=[df['HousePrice'].min(), boundary1, boundary2, df['HousePrice'].max()],
                                     labels=[0.2, 0.8, 1.0]).astype(float)

    # Aggregate to grid level
    grouped = df.groupby('HousePrice').agg({
        'II':'mean','ES':'mean','bypass':'mean','NNHI':'mean',
        'home_road_dist':'mean','road_closest':'mean','road_second':'mean','road_third':'mean',
        'HousePrice_10cut':'mean','HousePrice_262cut':'mean','HousePrice_3cut':'mean'
    }).reset_index()
    grouped.to_csv(PATH_RESULT3 / f"{CITY}_ES.csv", index=False, encoding='utf-8-sig')
    dfs.append(grouped)

# -------------------------
# Combine all cities
# -------------------------
data = pd.concat(dfs, axis=0)
data['ES'] = scale_to_0_10(data['ES'])
data['II'] = scale_to_0_10(data['II'])
data['bypass_10'] = pd.qcut(scale_to_0_1(data['bypass']), 10, labels=False, duplicates='drop') + 1
data['road_closest'] = pd.qcut(scale_to_0_1(data['road_closest']), 10, labels=False, duplicates='drop') + 1
data['HousePrice_262cut_grid'] = pd.qcut(data['HousePrice'], 3, labels=[0.2,0.8,1.0]).astype(float)
data.to_csv(OUTPUT_PATH / "ES_combined.csv", index=False, encoding='utf-8-sig')

# -------------------------
# Figure 4 plotting
# -------------------------
plt.rcParams['font.family'] = 'Arial'
Fontsize_ax = 27
Fontsize_legend = 26
Fontsize_label = 22
WIDTH = 0.4

custom_palette = {0.2:'#E69191', 0.8:'#CFE7C4', 1.0:'#92B5CA'}
custom_labels = ['Low-SES group','Moderate-SES group','High-SES group']

fig, axs = plt.subplots(2,2,figsize=(21.2,18.3), gridspec_kw={'hspace':0.16,'wspace':0.2})

for i, (feature, ax) in enumerate(zip(["ES","II","ES","II"], axs.flatten())):
    if i < 2:
        sns.boxplot(data=data, x="bypass_10", y=feature, hue="HousePrice_262cut_grid",
                    showfliers=False, legend=False, width=WIDTH, palette=custom_palette, ax=ax)
        ax.set_xlabel('Bypass level', fontsize=Fontsize_ax)
    else:
        sns.boxplot(data=data, x="road_closest", y=feature, hue="HousePrice_262cut_grid",
                    showfliers=False, legend=False, width=WIDTH, palette=custom_palette, ax=ax)
        ax.set_xlabel('Hospital accessibility', fontsize=Fontsize_ax)
    ax.set_ylabel('Experienced segregation index' if i%2==0 else 'Experienced socioeconomic class index',
                  fontsize=Fontsize_ax)
    ax.tick_params(axis='both', which='major', labelsize=Fontsize_label)

# Create legend
legend_handles = [mpatches.Patch(color=color, label=label)
                  for label, color in zip(custom_labels, custom_palette.values())]
fig.legend(handles=legend_handles, labels=custom_labels, fontsize=Fontsize_legend,
           bbox_to_anchor=(0.5,0.06), loc='upper center', ncol=len(custom_labels), frameon=False)

plt.tight_layout()
plt.savefig(OUTPUT_PATH / 'Figure4.jpg', dpi=600, bbox_inches='tight', pad_inches=0.1)
print("Figure 4 saved successfully!")
