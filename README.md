# Fused-Lasso Estimation of Structural Breaks in Panel Data with Interactive Effects

This repository accompanies the article

> **Kaddoura, Y., and J. Westerlund (2023).**  
> *Estimation of Panel Data Models with Random Interactive Effects and Multiple Structural Breaks when T is Fixed*.  
> **Journal of Business & Economic Statistics**, 41(3), 778–790.  
> DOI: [10.1080/07350015.2022.2067546](https://doi.org/10.1080/07350015.2022.2067546)

Simply some code for the above paper. 

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

The baseline parameters in `panel_breaks_fused_lasso.py` are

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

Tweek these as needed.


## The local package `TimeSeriesP`

The file `TimeSeriesP/lag.py` defines

```python
def Lag(n):
```

which returns the $n \times n$ lag operator matrix used to construct the lagged dependent variable in the dynamic design. Keeping it in a local package makes the main script cleaner and preserves the import style already used in the paper code.

