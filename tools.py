
# =========================================================
# BACKTEST
# =========================================================

import numpy as np
import pandas as pd

from model import solveHedgingModel, solveHedgingWithMoneyConstraint, solveStandardModel, solveWithMoneyConstraint

def run_strategy(returns, window, freq, BETA, MIN_RETURN, MAX_ALLOCATION, solver="gurobi"):
    n = returns.shape[1]
    weights = np.zeros(n)

    portfolio_returns = []
    weights_history = []

    for t in range(window, len(returns)):
        if freq == "daily" or (freq == "window" and (t - window) % window == 0):
            R_window = returns.iloc[t-window:t].values
            mean = R_window.mean(axis=0)

            weights, _, _ = solveStandardModel(
                R_window,
                mean,
                beta=BETA,
                minReturn=MIN_RETURN,
                maxAllocation=MAX_ALLOCATION,
                solver=solver
            )

        r = returns.iloc[t].values @ weights
        portfolio_returns.append(r)
        weights_history.append(weights.copy())

    return (
        pd.Series(portfolio_returns, index=returns.index[window:]),
        np.array(weights_history) 
    )

def run_with_money_constraint(returns, window, freq, MONEY_CONSTRAINT, BETA, MIN_RETURN, MAX_ALLOCATION, solver="gurobi"):
    n = returns.shape[1]
    weights = np.zeros(n)

    portfolio_returns = []
    weights_history = []

    for t in range(window, len(returns)):
        if freq == "daily" or (freq == "window" and (t - window) % window == 0):
            R_window = returns.iloc[t-window:t].values
            mean = R_window.mean(axis=0)

            weights, _, _ = solveWithMoneyConstraint(
                R_window,
                mean,
                beta=BETA,
                minReturn=MIN_RETURN,
                maxAllocation=MAX_ALLOCATION,
                moneyConstraint=MONEY_CONSTRAINT,
                solver=solver
            )

        r = returns.iloc[t].values @ weights
        portfolio_returns.append(r)
        weights_history.append(weights.copy())

    return (
        pd.Series(portfolio_returns, index=returns.index[window:]),
        np.array(weights_history) 
    )

def run_hedging_strategy(returns, window, freq, BETA, MIN_RETURN, MAX_ALLOCATION, solver="gurobi"):
    n = returns.shape[1]
    weights = np.zeros(n)

    portfolio_returns = []
    weights_history = []

    for t in range(window, len(returns)):
        if freq == "daily" or (freq == "window" and (t - window) % window == 0):
            R_window = returns.iloc[t-window:t].values
            mean = R_window.mean(axis=0)

            weights, _, _ = solveHedgingModel(
                R_window,
                mean,
                beta=BETA,
                minReturn=MIN_RETURN,
                maxAllocation=MAX_ALLOCATION,
                solver=solver
            )

        r = returns.iloc[t].values @ weights
        portfolio_returns.append(r)
        weights_history.append(weights.copy())

    return (
        pd.Series(portfolio_returns, index=returns.index[window:]),
        np.array(weights_history) 
    )

def run_hedging_with_money_constraint(returns, window, freq, MONEY_CONSTRAINT, BETA, MIN_RETURN, MAX_ALLOCATION, solver="gurobi"):
    n = returns.shape[1]
    weights = np.zeros(n)

    portfolio_returns = []
    weights_history = []

    for t in range(window, len(returns)):
        if freq == "daily" or (freq == "window" and (t - window) % window == 0):
            R_window = returns.iloc[t-window:t].values
            mean = R_window.mean(axis=0)

            weights, _, _ = solveHedgingWithMoneyConstraint(
                R_window,
                mean,
                beta=BETA,
                minReturn=MIN_RETURN,
                maxAllocation=MAX_ALLOCATION,
                moneyConstraint=MONEY_CONSTRAINT,
                solver=solver
            )

        r = returns.iloc[t].values @ weights
        portfolio_returns.append(r)
        weights_history.append(weights.copy())

    return (
        pd.Series(portfolio_returns, index=returns.index[window:]),
        np.array(weights_history) 
    )

# =========================================================
# METRICS
# =========================================================

def total_return(r):
    if isinstance(r, pd.DataFrame):
        r = r.iloc[:, 0]
    return float((1 + r).prod() - 1)
