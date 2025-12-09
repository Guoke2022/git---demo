# METHODS

This document provides methodological details for the computation of the Nth Nearest Hospital Index (NNHI), hospital accessibility (E2SFCA), the experienced segregation (ES) and experienced socioeconomic class (ESC) indices, and the discrete choice model used to characterize hospital choice behavior.

---

## 1. Nth Nearest Hospital Index (NNHI)

The Nth Nearest Hospital Index (NNHI) was introduced to depict patients' hospital bypass behaviours. NNHI is computed by sorting the road-network distances between a patient's residence and all hospitals in the city in ascending order, and identifying the rank of the actually visited hospital.

Formally, for patient \(i\), let \(d_{i1} \le d_{i2} \le \dots \le d_{iJ}\) denote the ordered road-network distances from their residence to all \(J\) hospitals. If the visited hospital corresponds to the \(N\)-th element in this ordered list, then the NNHI for patient \(i\) is:

\[
\text{NNHI}_i = N.
\]

This rank reflects the patient's willingness to prioritize more distant hospitals over closer ones. For example:

- \(\text{NNHI} = 1\): the patient visits the **nearest** hospital (non-bypass behaviour).
- \(\text{NNHI} = 5\): the patient visits the **5th nearest** hospital (bypass behaviour).

Higher NNHI values therefore indicate stronger bypass behaviour.

---

## 2. Hospital Accessibility (E2SFCA)

Hospital accessibility was evaluated using the Enhanced Two-Step Floating Catchment Area (E2SFCA) method. Hospital accessibility at demand location \(i\) is denoted as \(A_i\).

### 2.1. General formulation

Accessibility at demand point \(i\) is calculated as:

\[
A_i = \sum_{j:\, t_{ij} \le T_r} R_j \, W_r
\tag{1}
\]

where:

- \(R_j\) is the supply–demand ratio at supply (hospital) location \(j\),
- \(t_{ij}\) is the shortest travel time from demand point \(i\) to hospital \(j\),
- \(W_r\) is the distance-decay weight for travel-time band \(r\),
- \(T_r\) denotes the travel-time threshold(s) considered in band \(r\).

The supply–demand ratio \(R_j\) at hospital \(j\) is given by:

\[
R_j = \frac{S_j}{\sum_{k:\, t_{kj} \in T_r} D_k}
\]

where:

- \(S_j\) represents the service capacity at hospital \(j\) (e.g., number of beds),
- \(D_k\) represents the demand (e.g., population size) at demand location \(k\),
- the denominator sums over all demand locations \(k\) whose travel time \(t_{kj}\) to hospital \(j\) falls within the catchment threshold(s).

Substituting \(R_j\) into (1) yields:

\[
A_i
= \sum_{j:\, t_{ij} \le T_r}
\frac{S_j}{\sum_{k:\, t_{kj} \in T_r} D_k} \, W_r.
\]

### 2.2. Time thresholds and distance decay

Supply \(S_j\) is based on hospital bed capacity, and demand \(D_k\) is based on population size. Time thresholds of 15, 30, and 60 minutes were used, motivated by the “Golden Hour” rule and empirical research on healthcare accessibility.

To account for distance decay, we employed a Gaussian decay function:

\[
w(t) = e^{-t^2 / \beta},
\]

with \(\beta\) set to 440. For each travel-time band \(r\), the corresponding weight \(W_r\) is obtained by evaluating \(w(t)\) at representative travel times within that band.

Shortest travel times \(t_{ij}\) were determined using Chinese urban road design specifications and OpenStreetMap-based road-network data.

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
