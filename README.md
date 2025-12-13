# Hospital-Bypass

Source code for "Spatial and socioeconomic inequalities in hospital utilization".

This repository contains a collection of Python scripts for simulating, processing, and analyzing synthetic hospital visit trajectories.
All data used in this project are fully synthetic and generated solely for demonstration and reproducibility purposes.
No real or personal information is included.

## Python Version

**Python 3.12** is required for this project. All the required packages are listed in `requirements.txt`.

---

## Overview of Scripts (src/)

Below is a concise and standardized description of each script's purpose.

---

### 1. hospital_visits_identification.py

Processes large-scale synthetic trajectory data to extract potential *single-day hospital visits*.
Key steps include:

- Chunk-based loading of raw CSVs
- Coordinate cleaning and bounding-box filtering
- Spatial matching with hospital polygons
- Identifying possible hospital visits via residence duration

*This script is for demonstration only and does not use real mobility data.*

---

### 2. excluding_non-patient_users.py

Identifies and removes synthetic “hospital staff” from daily trajectory records based on long-term appearance frequency.
Workflow:

- Load multiple `single_day_patients_*.csv` files
- Flag users appearing ≥ N days as staff
- Save staff-ID list
- Output cleaned daily patient files

*Also based entirely on synthetic data.*

---

### 3. family_accompany_identification.py

Identifies family companions or accompanying persons among patients through spatiotemporal analysis. Implements a comprehensive multi-stage processing pipeline:

**Core Processing Steps:**

1. **Hospital Contact Detection** - Identifies patient pairs with contact at hospitals based on spatiotemporal overlap analysis
2. **Residence Contact Detection** - Detects patient pairs with contact at their residences based on geographic distance calculations
3. **ABID Pair Matching** - Matches patient pairs that satisfy both hospital contact and residence contact conditions
4. **Trajectory Record Extraction** - Extracts complete spatiotemporal trajectory records of matched patient pairs
5. **Journey Contact Detection** - Calculates patient contact situations outside hospitals and residential areas
6. **Companion Ratio Calculation** - Builds patient contact network graphs and classifies patient and companion roles

The specific functionality implementation of this step is located in `src/accompany_identification_core`.
*All data used in this step are also simulated data.*

---

### 4. figure1_analysis.py

Computes statistics and generates the visualizations used in **Figure 1** of the study.
Includes:

- Preprocessing
- Extra travel ratio calculations
- Visualizing travel distance, extra distance, and bypass rate for cities and city groups

---

### 5. figure2_analysis.py

Computes summary tables and visualization for **Figure 2**.
Includes:

- Summary CSVs for NNHI, extra travel distance, and bypass rate
- Subsets for top-tier and high-reputation hospitals
- Scatterplots (with log-fit) and bar charts

---

### 6. figure3_analysis.py

Generates **Figure 3**, including:

- Dual bar plots of bypass rate and NNHI by SES groups
- Lorenz curves
- C-index (CI) calculations for road distance, NNHI, and bypass

---

### 7. figure4_analysis.py

Computes:

- Experienced Segregation (ES)
- Income Index (II / ESC in manuscript)

Generates **Figure 4**:

- Boxplots of ES and income index across bypass groups and hospital accessibility levels

---

### 8. NNHI_calculate.py

Computes the **Nearest N Hospitals Index (NNHI)** and associated road-network distances.
Includes:

- NNHI calculations
- Road-network distance computation via OSMnx

Written for academic replication; does **not** require actual road networks or city datasets to run.

---

### 9. sensitivity_analysis.py

Performs sensitivity analysis for hospital bypass behavior and SES disparities.
Includes:

- Sensitivity testing using **alternative bypass metrics**
- Sensitivity testing for **patient identification thresholds**
Outputs:
- CSV tables for robustness checks
- City-level and group-level summary results

---

### 10. MixedLogit

Performs mixed logit–based discrete choice modeling to analyze hospital bypass behavior.

Includes:

- Estimation of trade-offs between hospital quality and travel distance
- Evaluation of systematic preference differences across SES groups
- Full workflow and methodological details (see METHODS.md)

Outputs:

- Estimated model parameters
- Overall-sample results (no SES grouping)
- SES-stratified results (High-SES, Middle-SES, Low-SES)

Files:

- step1_process.py – Data preprocessing for constructing patient–hospital choice sets
- step2_mixlogit.py – Mixed Logit model estimation
- MixedLogit_without_interaction_no_group – Results for the overall sample
- MixedLogit_without_interaction_SES_group – Results for SES groups

---

## Methods

We also provide an additional documentation file (**METHODS.md**) that describes the analytical methods used in this project in detail. The document includes:

- The algorithm for computing the **Nth Nearest Hospital Index (NNHI)**
- The procedure for calculating **hospital accessibility** using the Enhanced Two-Step Floating Catchment Area (**E2SFCA**) method
- The methodology for estimating the **experienced segregation (ES)** index
These methodological explanations are intended to support reproducibility, transparency, and academic use of the accompanying scripts.

---

## Requirements

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

The main dependencies include:

- pandas==2.3.3
- numpy==2.3.4
- geopandas==1.1.1
- haversine==2.9.0
- osmnx==2.0.7
- shapely==2.1.2
- seaborn==0.12.2
- matplotlib==3.9.0
- scipy==1.16.3
- xlogit==0.2.7
- cupy-cuda12x==13.6.0
- polars==1.31.0
- scikit-learn==1.7.2
- networkx==3.6.1
- tqdm==4.67.1
