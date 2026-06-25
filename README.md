# Risk Aversion Decision Making for Portfolio Optimization

## Overview

This project investigates **risk-averse portfolio optimization** using a **Conditional Value at Risk (CVaR)** framework. The objective is to construct portfolios that minimize downside risk while maintaining diversification across a set of financial assets.

Using historical market data, the system repeatedly solves a CVaR optimization model over rolling windows and evaluates portfolio performance under different rebalancing strategies. Hyperparameter combinations are tested on a training period and the best-performing strategies are then evaluated on an out-of-sample test period.

---

## Background

Portfolio management seeks to balance expected returns against exposure to risk. Traditional approaches often focus on variance-based measures of risk; however, variance penalizes both positive and negative deviations from the mean.

To better capture downside exposure, this project utilizes **Conditional Value at Risk (CVaR)**, a metric that estimates the expected loss occurring in the worst portion of possible outcomes.

### CVaR Optimization Model

Given a portfolio allocation vector:

$$
X = (x_1, x_2, ..., x_n)
$$

and asset returns:

$$
R_{t,i}
$$

portfolio loss at time \(t\) is defined as:

$$
L_t(X) = - \sum_{i=1}^{n} R_{t,i}x_i
$$

The CVaR optimization problem is:

$$
\min \; \alpha + \frac{1}{(1-\beta)T}\sum_{t=1}^{T}u_t
$$

Subject to:

$$
\sum_{i=1}^{n}x_i = 1
$$

$$
x_i \ge 0
$$

$$
u_t \ge L_t(X)-\alpha
$$

$$
u_t \ge 0
$$

where:

- \(X\) = portfolio weights
- \(\alpha\) = Value-at-Risk (VaR) threshold
- \(u_t\) = auxiliary variables measuring losses beyond VaR
- \(\beta\) = confidence level

This formulation minimizes expected losses in the worst \(1-\beta\) fraction of historical outcomes.

---

## Assets

The portfolio is constructed from the following securities:

| Ticker | Asset |
|----------|----------|
| GLD | SPDR Gold Shares |
| BND | Vanguard Total Bond Market ETF |
| IGV | iShares Expanded Tech-Software Sector ETF |
| VDC | Vanguard Consumer Staples ETF |
| SOCL | Global X Social Media ETF |
| IHF | iShares U.S. Healthcare Providers ETF |
| IYZ | iShares U.S. Telecommunications ETF |
| XOM | Exxon Mobil Corporation |

### Benchmark

The benchmark portfolio is:

- **SPY** (S&P 500 ETF)

---

## Dataset

Historical prices are downloaded using Yahoo Finance via `yfinance`.

### Date Ranges

| Dataset | Range |
|----------|----------|
| Full Data | 2023-01-01 → 2026-01-01 |
| Training | 2023-01-01 → 2024-12-31 |
| Testing | 2025-01-01 → 2026-01-01 |

Returns are computed using:

$$
R_t = \frac{P_t-P_{t-1}}{P_{t-1}}
$$

where:

- \(P_t\) is the adjusted closing price at time \(t\)

---

## Hyperparameter Search

The experiment evaluates multiple combinations of rolling-window sizes and portfolio rebalancing frequencies.

### Window Sizes

```python
WINDOWS = list(range(7, 91, 7))
```

Evaluated windows:

```text
7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84
```

### Rebalancing Frequencies

```python
FREQS = ["window", "daily"]
```

#### Daily

Portfolio weights are recomputed every trading day.

#### Window

Portfolio weights are recomputed once per optimization window.

### Risk Parameters

```python
BETA = 0.95
MIN_RETURN = 0.0
MAX_ALLOCATION = 1.0
```

| Parameter | Description |
|------------|------------|
| BETA | CVaR confidence level |
| MIN_RETURN | Minimum target return |
| MAX_ALLOCATION | Maximum allocation permitted in a single asset |

---

## Methodology

### 1. Download Historical Data

Price data for all portfolio assets and the benchmark are retrieved using Yahoo Finance.

### 2. Compute Returns

Daily percentage returns are calculated:

```python
returns = prices.pct_change().dropna()
```

### 3. Split Data

The data is partitioned into:

- Training period
- Test period

### 4. Hyperparameter Search

For every combination of:

- Window size
- Rebalancing frequency

the portfolio strategy is executed on the training dataset.

Performance is measured by cumulative portfolio growth.

### 5. Select Top Strategies

The top-performing configurations are ranked by total return.

```python
TOP_N = 3
```

### 6. Out-of-Sample Testing

The selected strategies are evaluated on unseen market data from the test period.

### 7. Benchmark Comparison

Results are compared against the SPY benchmark.

---

## Parallel Processing

Hyperparameter evaluations are parallelized using Python multiprocessing:

```python
with mp.Pool(mp.cpu_count()) as pool:
```

This allows all available CPU cores to participate in the search process.

---

## Output Files

All experiment artifacts are stored in:

```text
exp_results/exp1/
```

### Training Performance

**File**

```text
exp1TrainingPerformance.csv
```

Contains:

| Column |
|----------|
| window |
| freq |
| growth |

Each row represents a unique hyperparameter configuration evaluated on the training set.

---

### Top Strategies

**File**

```text
exp1TopPortfolios.csv
```

Stores the highest-performing parameter combinations identified during training.

---

### Test Allocations

**File**

```text
exp1TestPerformance.csv
```

Contains all portfolio allocations generated during the testing period.

Columns:

| Column |
|----------|
| window |
| freq |
| count |
| growth |
| GLD |
| BND |
| IGV |
| VDC |
| SOCL |
| IHF |
| IYZ |
| XOM |

Each row corresponds to one portfolio solution produced during rebalancing.

---

### Performance Visualization

**File**

```text
exp1Plot.png
```

Displays cumulative returns for:

- Top CVaR strategies
- SPY benchmark

allowing direct visual comparison of performance.

---

## Dependencies

Install required packages:

```bash
pip install yfinance pandas matplotlib tqdm scipy
```

Depending on the implementation of `run_strategy`, additional optimization libraries may be required.


---

## Running the Experiment

Execute:

```bash
python experiment.py
```

The workflow will:

1. Download historical market data.
2. Perform a parallel hyperparameter search.
3. Rank portfolio configurations.
4. Evaluate the top strategies on unseen data.
5. Compare results against SPY.
6. Generate CSV reports and plots.

---

## Performance Metric

Portfolio growth is computed using:

```python
total_return(strategy_returns)
```

which evaluates:

$$
\text{Growth} = \prod_{t=1}^{T}(1+r_t)-1
$$

where:

- \(r_t\) is the portfolio return at time \(t\)

Higher growth indicates stronger overall performance.

---

## Research Objective

The goal of this study is to evaluate whether **CVaR-based portfolio optimization** can produce superior risk-adjusted performance compared to a passive market benchmark.

Specifically, the experiments investigate the impact of:

- Lookback window size
- Portfolio rebalancing frequency
- Downside-risk minimization

on out-of-sample portfolio performance under changing market conditions.

## Contributors
- Jeremy Perez
- Janak Subedi
