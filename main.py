ampl_model = """
param T;
param n;
param beta;

param R{1..T, 1..n};
param mu{1..n};
param r_min;
param w_max;

var x{1..n} >= 0, <= w_max;
var alpha;
var u{1..T} >= 0;

minimize CVaR:
    alpha + (1 / ((1 - beta) * T)) * sum {t in 1..T} u[t];

subject to TailRisk {t in 1..T}:
    u[t] >= -sum {i in 1..n} R[t,i] * x[i] - alpha;

subject to Budget:
    sum {i in 1..n} x[i] = 1;

subject to ReturnTarget:
    sum {i in 1..n} mu[i] * x[i] >= r_min;
"""

# import ampl
import numpy as np
from amplpy import AMPL

def solve_cvar_ampl(R, mu, beta=0.95, r_min=0.0, w_max=0.2):
    """
    Solve single CVaR portfolio optimization using AMPL in Python.
   
    Returns:
        x (np.array): optimal portfolio weights
        alpha (float)
        obj (float)
    """

    T, n = R.shape

    ampl = AMPL()

    # -------------------------
    # Load model
    # -------------------------
    ampl.eval(ampl_model)

    # -------------------------
    # Set parameters
    # -------------------------
    # ampl.set["T"] = range(1, T + 1)
    # ampl.set["n"] = range(1, n + 1)
    ampl.param["T"] = T
    ampl.param["n"] = n

    ampl.param["beta"] = beta
    ampl.param["r_min"] = r_min
    ampl.param["w_max"] = w_max

    # R matrix
    for t in range(T):
        for i in range(n):
            ampl.param["R"][t + 1, i + 1] = float(R[t, i])

    # mu vector
    for i in range(n):
        ampl.param["mu"][i + 1] = float(mu[i])

    # -------------------------
    # Solve
    # -------------------------
    # ampl.solve()
    ampl.option["solver"] = "highs"
    ampl.solve()

    # -------------------------
    # Extract results
    # -------------------------
    x = np.array([ampl.var["x"][i + 1].value() for i in range(n)])
    alpha = ampl.var["alpha"].value()
    obj = ampl.getObjective("CVaR").value()

    return x, alpha, obj


#------------------------------------------------------------------------------
#YFINAance data loading and backtesting code
#------------------------------------------------------------------------------

import yfinance as yf

# -------------------------
# Load data
# -------------------------
tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

prices = yf.download(tickers, start="2025-01-01", end="2026-01-01", auto_adjust=True)["Close"]
returns = prices.pct_change().dropna()

R = returns.values
mu = R.mean(axis=0)

# -------------------------
# Solve CVaR
# -------------------------
x, alpha, obj = solve_cvar_ampl(
    R,
    mu,
    beta=0.95,
    r_min=0.0005,
    w_max=1.0,
)

# -------------------------
# Output
# -------------------------
print("Optimal weights:")
for i, t in enumerate(tickers):
    print(f"{t}: {x[i]:.4f}")

print("\nAlpha (VaR):", alpha)
print("Objective (CVaR):", obj)

