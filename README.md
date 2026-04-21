# Fused-Lasso Estimation of Structural Breaks in Panel Data with Interactive Effects

This repository accompanies the article

> **Kaddoura, Y., and J. Westerlund (2023).**  
> *Estimation of Panel Data Models with Random Interactive Effects and Multiple Structural Breaks when T is Fixed*.  
> **Journal of Business & Economic Statistics**, 41(3), 778–790.  
> DOI: [10.1080/07350015.2022.2067546](https://doi.org/10.1080/07350015.2022.2067546)

The code implements a fused-lasso approach to estimating structural breaks in short panels with interactive effects. It is written as a research companion to the paper rather than as a general-purpose software library.

## Overview

The paper studies the panel model

$$
y_{it} = x_{it}'\beta_t + u_{it}, \qquad
u_{it} = \lambda_i' f_t + \varepsilon_{it},
\qquad i=1,\dots,N,\; t=1,\dots,T.
$$

The coefficient path $\{\beta_t\}_{t=1}^T$ is allowed to change across regimes, while unobserved interactive effects remain present in the error term. Break detection is cast as a shrinkage problem: adjacent coefficient vectors are penalized so that the estimated path is piecewise constant. In stylized form, the estimator solves

$$
\min_{\beta_1,\dots,\beta_T}
\frac{1}{NT}\sum_{t=1}^T \lVert y_t - X_t \beta_t \rVert_2^2
\;+\;
\lambda \sum_{t=2}^T \lVert \beta_t - \beta_{t-1} \rVert_2 .
$$

## Installation

### Conda

```bash
conda env create -f environment.yml
conda activate fused-lasso-jbes
```

### Pip

```bash
pip install -r requirements.txt
```

## Running the script

The current simulation file is

```bash
python panel_breaks_fused_lasso.py
```

The script is written as a self-contained Monte Carlo program. At the bottom of the file, the user sets the simulation dimensions, the grid of penalty values, and the number of replications.

## Parameters in the simulation script

The baseline parameter block in `panel_breaks_fused_lasso.py` is

```python
p   = 4
n   = 25
T   = 5
m   = 1
r   = 5
```

together with

```python
lambd_values = np.logspace(-3, 3, 50)
sim = 1000
```

and, in the `DATA3` design, additional serial and cross-sectional dependence parameters:

```python
phi
phi_1
pi
```

The role of each quantity is:

| Parameter | Meaning |
|---|---|
| `p` | Number of regressors before the lagged dependent variable is appended in the dynamic design. |
| `n` | Number of cross-sectional units. |
| `T` | Number of time periods in the original panel. |
| `m` | Number of true structural breaks in the data-generating process. |
| `r` | Number of latent factors in the interactive-effects structure. |
| `phi` | Persistence parameter in the factor process used in `DATA3`. |
| `phi_1` | Serial and local cross-sectional dependence parameter used in the idiosyncratic error process in `DATA3`. |
| `pi` | Persistence / dependence parameter used in the regressor innovation process in `DATA3`. |
| `lambd_values` | Grid of candidate tuning parameters for the fused-lasso penalty. |
| `sim` | Number of Monte Carlo replications. |

## Data-generating processes

The script defines three DGP classes.

### `DATA1`

`DATA1` generates panel data without interactive effects. The methods correspond to different break structures:

- `DGP1()` : one break
- `DGP2()` : two breaks
- `DGPA()` : break at every date
- `DGPO()` : no break

### `DATA2`

`DATA2` augments the model with random interactive effects. It returns, among other objects, factor loadings, factors, demeaned outcomes, and demeaned regressors.

### `DATA3`

`DATA3` is the richest design. It allows for

- dynamic dependence through a lagged dependent variable,
- interactive effects,
- serially dependent idiosyncratic errors,
- cross-sectional spillovers in both the error term and regressor innovations.

Because the lagged dependent variable is added to the regressor matrix, the effective regressor dimension becomes `p + 1` inside this design.

## Optimization routines

The class `Optimize` contains three estimation routines:

- `OLS(X, y)`  
  Date-by-date least-squares benchmark.

- `FGLS(X, y, b_o, Lambda)`  
  Penalized estimator with fused-lasso type shrinkage across adjacent dates. The penalty is adaptively weighted using a preliminary estimate `b_o`.

- `NBOLS(X, y)`  
  Pooled no-break least-squares benchmark.

## Tuning-parameter selection

The function `IC(lambd_values, y, X, p, T, n)` loops over the penalty grid and evaluates an information criterion of the form

$$
IC(\lambda)
=
\frac{1}{NT}\sum_{t=1}^T \left\| y_t - X_t \widehat\beta_t(\lambda) \right\|_2^2
+
\omega_{NT}\, p\,(\widehat m_\lambda + 1),
$$

where $\widehat m_\lambda$ is the estimated number of breaks and the script uses

$$
\omega_{NT} = \frac{\log(NT)}{\sqrt{NT}}.
$$

The selected tuning parameter is the value of $\lambda$ that minimizes this criterion over the user-specified grid.

## The local package `TimeSeriesP`

The file `TimeSeriesP/lag.py` defines

```python
def Lag(n):
```

which returns the $n \times n$ lag operator matrix used to construct the lagged dependent variable in the dynamic design. Keeping it in a local package makes the main script cleaner and preserves the import style already used in the paper code.

