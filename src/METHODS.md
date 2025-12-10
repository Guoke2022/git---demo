# METHODS

This document provides methodological details for the computation of the Nth Nearest Hospital Index (NNHI), hospital accessibility (E2SFCA), the experienced segregation (ES) and experienced socioeconomic class (ESC) indices, and the discrete choice model used to characterize hospital choice behavior.

---

## 1. Nth Nearest Hospital Index (NNHI)

The Nth Nearest Hospital Index (NNHI) was used to characterize hospital bypass behaviour. NNHI measures how far down the ordered list of nearby hospitals a patient chooses to visit.

To compute NNHI for each patient:

Calculate the road network distance from the patient’s residence to all hospitals in the city.

Sort these distances in ascending order.

Identify the rank of the hospital actually visited.

Record this rank as the NNHI value.

Interpretation:

NNHI = 1: The patient visited the nearest hospital (no bypass behaviour).

NNHI > 1: The patient bypassed closer hospitals and chose a more distant one.

For example, NNHI = 5 indicates that the patient selected the fifth nearest hospital.

This metric provides a simple, interpretable indicator of patients' preferences or constraints that lead them to bypass nearby facilities.

---

## 2. Hospital Accessibility (E2SFCA)

Hospital accessibility was evaluated using the Enhanced Two-Step Floating Catchment Area (E2SFCA) method. The accessibility score for demand location i, denoted as $A_i$, is defined as:

![alt text](../images/E2SFCA.png)

where:

$R_j$ is the supply-to-demand ratio at hospital $j$

$t_{ij}$ is the shortest travel time between demand point $i$ and hospital $j$

$W_r$ is the distance-decay weight corresponding to travel-time zone $r$

Supply ($S_j$) is measured by hospital bed capacity

Demand ($D_k$) is measured by the population size at demand location $k$

Travel-time thresholds of 15, 30, and 60 minutes were adopted based on the “Golden Hour” principle as well as extensive prior research on healthcare accessibility.

Distance-decay weights $W_r$ were derived from a Gaussian decay function:

![alt text](E2SFCA2.png)

with $\beta = 440$.

Shortest-path travel times were calculated using OpenStreetMap road network data combined with Chinese urban road design specifications.

---

## 3. Experienced Segregation (ES) and Experienced Socioeconomic Class (ESC)

To analyse the impact of hospital bypass behaviour on social segregation, we adopted the experienced segregation (ES) index as proposed in earlier literature. This index quantifies the degree to which patients from different economic backgrounds effectively share hospital services.

### 3.1. Income rank assignment

Within each city, residential units (200 m × 200 m grids) are sorted into deciles based on their mean residential property value. These decile ranks are then assigned to patients residing in each unit, providing an income rank \(r_i\) for patients originating from unit \(i\).

### 3.2. Experienced integration at the hospital level

For individuals from unit \(i\) accessing hospital \(L\), the **experienced integration** at hospital \(L\) is defined as:

\[
\text{experienced\_integration}_{(i,L)} =
\frac{1}{n - 1}
\sum_{j \neq i}^{n}
\left| r_{(i,L)} - r_{(j,L)} \right|
\tag{2}
\]

where:

- \(r_{(j,L)}\) is the income rank of individuals from unit \(j\) who visit hospital \(L\),
- \(n\) is the total number of distinct origin units whose residents access hospital \(L\),
- \(\left| r_{(i,L)} - r_{(j,L)} \right|\) is the absolute income-rank difference between units \(i\) and \(j\) among patients visiting the same hospital.

This measures the degree of interaction or co-presence between patients from unit \(i\) and patients from other income ranks at hospital \(L\).

### 3.3. Aggregating experienced integration to the unit level

We then aggregate over all hospitals visited by residents of unit \(i\) to obtain a unit-level measure of experienced integration:

\[
\text{experienced\_integration}_i =
\frac{
\sum_{L \in \text{HOSPs}}
\text{experienced\_integration}_{(i,L)} \cdot p_{(i,L)}
}{
\sum_{L \in \text{HOSPs}} p_{(i,L)}
}
\tag{3}
\]

where:

- \(p_{(i,L)}\) is the number of people from unit \(i\) who visit hospital \(L\),
- the summation runs over all hospitals \(L\) accessed by individuals from unit \(i\).

### 3.4. Experienced segregation index

The **experienced segregation** for unit \(i\) is then defined as:

\[
\text{experienced\_segregation}_i =
1 - \text{experienced\_integration}_i.
\tag{4}
\]

Higher values of \(\text{experienced\_segregation}_i\) indicate stronger effective segregation in hospital use by residents of unit \(i\).

### 3.5. Experienced socioeconomic class (ESC)

To determine whether a hospital is predominantly accessed by low- or high-property-value communities, we define an experienced socioeconomic class (ESC) index based on the average income rank of patients accessing each hospital.

For individuals from unit \(i\) accessing hospital \(L\), we compute:

\[
\text{experienced\_socioeconomic\_class}_{(i,L)} =
\frac{1}{n} \sum_{j=1}^{n} r_{(j,L)}
\tag{5}
\]

where:

- \(r_{(j,L)}\) is the income rank of individuals from unit \(j\) who visit hospital \(L\),
- \(n\) is the total number of origin units whose residents visit hospital \(L\).

This represents the average socioeconomic status of the patient mix at hospital \(L\), experienced by individuals from unit \(i\).

Aggregating to the unit level, we obtain:

\[
\text{experienced\_socioeconomic\_class}_i =
\frac{
\sum_{L \in \text{HOSPs}}
\text{experienced\_socioeconomic\_class}_{(i,L)} \cdot p_{(i,L)}
}{
\sum_{L \in \text{HOSPs}} p_{(i,L)}
}
\tag{6}
\]

where \(p_{(i,L)}\) is again the number of people from unit \(i\) who visit hospital \(L\).

Finally, both ES and ESC measures are scaled to range between 0 and 10, where 10 corresponds to the highest level (e.g., highest segregation or highest experienced socioeconomic class) and 0 to the lowest.

---

## 4. Discrete Choice Model

To characterize hospital choice behaviour, we adopt a mixed logit (random-coefficients logit) model capturing the trade-off between hospital quality and travel distance.

### 4.1. Utility specification

The utility that patient \(i\) derives from choosing hospital \(j\) is specified as:

\[
U_{ij}
=
\beta_G G_j
+
\beta_B B_j
+
\beta_R R_j
-
\left(
\beta_D^1 D_{ij}
+
\beta_D^2 D_{ij}^2
+
\beta_D^3 D_{ij}^3
\right)
+
\varepsilon_{ij}
\tag{7}
\]

where:

- \(G_j\): hospital grade indicator (e.g., Tertiary Grade A, secondary hospital),
- \(B_j\): hospital bed capacity (in hundreds of beds),
- \(R_j\): hospital reputation score (e.g., Fudan Chinese Hospital ranking),
- \(D_{ij}\): travel distance from patient \(i\)'s residence to hospital \(j\) (measured in 100 km units),
- \(\beta_G, \beta_B, \beta_R\): coefficients for hospital quality attributes,
- \(\beta_D^1, \beta_D^2, \beta_D^3\): coefficients for the cubic distance specification,
- \(\varepsilon_{ij}\): idiosyncratic error term.

The cubic distance specification allows for flexible, non-linear distance decay in hospital choice. The error term \(\varepsilon_{ij}\) is assumed to follow a Type I extreme value distribution.

To capture preference heterogeneity across patients, coefficients are allowed to vary randomly in the population (mixed logit specification), and both mean preferences and standard deviations are estimated.

Given computational constraints, the model is estimated via maximum simulated likelihood using an 8% random sample of patients. We estimate both pooled models and SES-stratified models to investigate systematic preference differences across socioeconomic groups.

### 4.2. Willingness to travel (WTT)

We compute the **willingness to travel (WTT)** as the marginal rate of substitution between hospital quality attributes and distance, evaluated at selected percentiles of the empirical distance distribution. WTT represents the additional distance patients are willing to travel for a one-unit improvement in hospital quality (e.g., upgrading from a secondary to a tertiary hospital).

For a given quality attribute \(Q\) (where \( \beta_Q \in \{\beta_G, \beta_B, \beta_R\} \)), WTT is computed as:

\[
\text{WTT}_{ij}
=
- \frac{\beta_Q}{
\beta_D^1
+
2 \beta_D^2 D_{\text{perc}}
+
3 \beta_D^3 D_{\text{perc}}^2
}
\tag{8}
\]

where:

- \(\beta_Q\) is the marginal utility coefficient associated with a quality attribute (grade, beds, or reputation),
- \(D_{\text{perc}}\) is a selected percentile of the empirical travel distance distribution (e.g., sampled at 5% intervals).

This provides a distance-equivalent interpretation of the marginal utility of hospital quality.

---
