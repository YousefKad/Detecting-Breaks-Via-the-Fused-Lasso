# Detecting Breaks via the Fused Lasso

Companion repository for

**Yousef Kaddoura** and **Joakim Westerlund**  
*Estimation of Panel Data Models with Random Interactive Effects and Multiple Structural Breaks when \(T\) is Fixed*  
**Journal of Business & Economic Statistics**, **41**(3), 778--790, 2023.  
DOI: `10.1080/07350015.2022.2067546`

---

## Overview

This repository documents a fused-lasso approach to structural break detection in panel data models with fixed \(T\) and large \(N\). The code in this repository is organized as a compact research companion rather than a general-purpose software library: the main script implements the data-generating processes, the optimization routines, the information criterion used to choose the tuning parameter, and the Monte Carlo experiment.

At a high level, the paper studies a panel model in which the slope coefficients are allowed to shift across latent regimes while unobserved interactive effects remain present. A representative specification is

\[
y_{it} = x_{it}'\beta_t + \lambda_i'F_t + \varepsilon_{it},
\qquad i = 1,\dots,N,\quad t = 1,\dots,T,
\]

where the coefficient path \(\{\beta_t\}_{t=1}^T\) is piecewise constant. Break detection is framed as a shrinkage problem by penalizing adjacent changes in the coefficient path. In stylized form, the estimator solves

\[
\min_{\beta_1,\dots,\beta_T}
\frac{1}{NT}\sum_{t=1}^T \lVert y_t - X_t\beta_t \rVert_2^2
\;+\;
\lambda \sum_{t=2}^T \lVert \beta_t - \beta_{t-1} \rVert_2.
\]

The implementation in this repository follows that logic and then combines it with an information-criterion step to choose the tuning parameter.

## Repository layout

```text
.
├── README.md
├── CITATION.cff
├── environment.yml
├── requirements.txt
├── monte_carlo_jbes.py
├── TimeSeriesP/
│   ├── __init__.py
│   └── lag.py
├── docs/
│   └── REPRODUCIBILITY_NOTES.md
└── paper/
    └── PAPER_METADATA.md
```

## Main script

The main script is:

```python
monte_carlo_jbes.py
```

It contains four main building blocks.

### 1. Data-generating processes

The code defines three DGP classes.

- `DATA1`: panel model without interactive effects.
- `DATA2`: panel model with interactive effects and cross-sectional demeaning.
- `DATA3`: richer dynamic design with lagged dependent variables, common factors, serial dependence, and cross-sectional dependence.

Each class provides methods such as:

- `DGP1()` for one break,
- `DGP2()` for two breaks,
- `DGPA()` for a break at every date,
- `DGPO()` for no breaks.

### 2. Optimization routines

The class `Optimize` contains:

- `OLS(...)`: date-by-date least squares benchmark,
- `FGLS(...)`: fused-lasso penalized estimation,
- `NBOLS(...)`: pooled no-break OLS benchmark.

### 3. Information criterion

The function

```python
IC(lambd_values, y, X, p, T, n)
```

loops over candidate penalties, estimates the fused-lasso solution, counts the number of detected breaks, and chooses the tuning parameter that minimizes the criterion.

### 4. Monte Carlo experiment

The script ends with a Monte Carlo block that repeatedly:

1. simulates a panel under the chosen DGP,
2. selects the tuning parameter,
3. estimates the coefficient path,
4. counts break-selection mistakes,
5. records average coefficient estimation error.

---

## Parameter guide

The script is controlled by a small set of core parameters.

### Global simulation parameters

| Parameter | Meaning |
|---|---|
| `p` | Number of regressors in the structural equation before adding the lagged dependent variable. |
| `n` | Number of cross-sectional units. |
| `T` | Number of time periods in the raw DGP. |
| `m` | Number of true structural breaks. |
| `r` | Number of latent common factors. |
| `sim` | Number of Monte Carlo replications. |
| `lambd_values` | Grid of candidate penalty values used by the information criterion. |

### Dependence parameters in `DATA3`

| Parameter | Meaning |
|---|---|
| `phi` | Persistence in the common factor process \(F_t\). |
| `phi_1` | Persistence / local dependence in the idiosyncratic component. |
| `pi` | Persistence / local dependence in the regressor innovation. |

### Function and class arguments

#### `DATA1(m, T, n, p)`

Use this class when simulating a panel without interactive effects.

#### `DATA2(r, m, T, n, p)`

Use this class when simulating interactive effects with latent factors and cross-sectional demeaning.

#### `DATA3(r, m, T, n, p, phi, phi_1, pi)`

Use this class when simulating the richer dynamic panel design with lagged outcomes and dependence in both regressors and disturbances.

#### `Optimize(p, T, n)`

Creates an estimator object for a problem of dimension \((p, T, n)\).

#### `IC(lambd_values, y, X, p, T, n)`

Selects the penalty parameter by minimizing the information criterion over the supplied grid.

---

## The `TimeSeriesP` package

This repository includes a lightweight local package:

```python
from TimeSeriesP.lag import Lag
```

The function `Lag(n)` returns the standard lag operator matrix used to construct lagged panel outcomes. Keeping it in a small package avoids hard-coded relative paths and makes the repository easier to run from the project root.

Formally, `Lag(n)` builds the matrix \(L\) such that post-multiplication generates the one-period lag in time-indexed arrays of length \(n\).

---

## Installation

### Conda environment

```bash
conda env create -f environment.yml
conda activate fused-lasso-breaks
```

### Pip alternative

```bash
pip install -r requirements.txt
```

---

## Running the code

From the project root, run:

```bash
python monte_carlo_jbes.py
```

The script prints the true coefficient path, the average estimated coefficient path, average detected breaks, false break frequencies, and a mean Frobenius error summary.

---

## Suggested workflow for replication

1. Start from the baseline configuration in `monte_carlo_jbes.py`.
2. Choose the DGP (`DATA1`, `DATA2`, or `DATA3`) that matches the experiment of interest.
3. Adjust the dimensions \((N, T, p, r)\) and break count \(m\).
4. Adjust the penalty grid `lambd_values`.
5. Increase `sim` once the design is stable.
6. Save tables and figures in a dedicated output folder if a replication archive is being prepared.

---

## Reproducibility notes

This repository is intended as a research companion. For a public archival version, the following additions are recommended:

- fixed random seeds for each experiment,
- explicit output directories for tables and figures,
- separate scripts for each design in the Monte Carlo section,
- a short note mapping each code block to the corresponding table or experiment in the paper.

A starting note is included in `docs/REPRODUCIBILITY_NOTES.md`.

---

## Citation

If you use this repository, please cite the paper:

```bibtex
@article{kaddoura_westerlund_2023,
  author  = {Kaddoura, Yousef and Westerlund, Joakim},
  title   = {Estimation of Panel Data Models with Random Interactive Effects and Multiple Structural Breaks when T is Fixed},
  journal = {Journal of Business \& Economic Statistics},
  year    = {2023},
  volume  = {41},
  number  = {3},
  pages   = {778--790},
  doi     = {10.1080/07350015.2022.2067546}
}
```

For software citation metadata, see `CITATION.cff`.
