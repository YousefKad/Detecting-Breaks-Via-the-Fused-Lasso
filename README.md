# Detecting Structural Breaks via the Fused Lasso

## Methodology

This repository implements the fused lasso approach for structural break detection in panel data models, as developed in our paper of Kaddoura and Westerlund (2024). The method provides a robust framework for identifying multiple structural breaks in economic and financial time series.

### Mathematical Foundation

The fused lasso estimator solves the following optimization problem:

**Objective Function:**

$$\ell_{\gamma}(\mathbb{B}_T) := \frac{1}{N} \sum_{i=1}^N\sum_{t = 1}^T (\tilde{\mathbb{y}}_{i,t} - \tilde{\mathbb{{x}}}_{i,t}'\mathbb{\beta}_t)^2
    + \gamma \cdot \sum_{t=2}^T w _t\|\mathbb{\beta}_{t}-\mathbb{\beta}_{t-1}\|$$

Where:
- $\tilde{y}_{it}$ is the observed dependent variable at time t and cross-sectional unit i
- $\tilde{x}_{it}$ is the observed  regressors at time t and cross-sectional unit i
- $\beta_t$ is the estimated slope coefficients at time t
- $w_t$ is an adaptive weight as given in the main paper.
- $\lambda$ is the regularization parameter controlling sparsity in differences
- $T$ is the total number of observations

### Key Features

The method offers several advantages:

1. **Multiple Break Detection**: Automatically identifies multiple structural breaks without prior knowledge of their number.
2. **Adaptive Regularization**: Employs data-driven selection of the regularization parameter λ using an IC function.
3. **Statistical Consistency**: Provides theoretically justified break point estimates.
4. **Computational Efficiency**: Implements optimized algorithms for large-scale time series.
