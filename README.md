# Detecting Structural Breaks via the Fused Lasso

## Methodology

This repository implements the fused lasso approach for structural break detection in time series, as developed in our paper. The method provides a robust framework for identifying multiple structural breaks in economic and financial time series.

### Mathematical Foundation

The fused lasso estimator solves the following optimization problem:

**Objective Function:**

minimize<sub>β</sub> ½∑<sub>t=1</sub><sup>T</sup> (y<sub>t</sub> - β<sub>t</sub>)² + λ∑<sub>t=2</sub><sup>T</sup> |β<sub>t</sub> - β<sub>t-1</sub>|

Where:
- y<sub>t</sub> is the observed time series at time t
- β<sub>t</sub> is the estimated level at time t
- λ is the regularization parameter controlling sparsity in differences
- T is the total number of observations

### Key Features

The method offers several advantages:

1. **Multiple Break Detection**: Automatically identifies multiple structural breaks without prior knowledge of their number
2. **Adaptive Regularization**: Employs data-driven selection of the regularization parameter λ
3. **Statistical Consistency**: Provides theoretically justified break point estimates
4. **Computational Efficiency**: Implements optimized algorithms for large-scale time series
