# Introduction to the methods


## Discrete Choice Model

To characterize hospital choice behaviour, we adopt a mixed logit (random-coefficients logit) model capturing the trade-off between hospital quality and travel distance.

### Utility specification

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

### Willingness to travel (WTT)

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

# Introduction to the code and file

## step1_process.py
- Data preprocessing.
## step2_mixlogit.py
- Discrete Mixed Logit Model Estimation.
## MixedLogit_without_interaction_no_group
The result of the no group.
## MixedLogit_without_interaction_SES_group
The result of the group.
