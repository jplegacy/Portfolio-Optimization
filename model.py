import numpy as np
from amplpy import AMPL


standardModel = """
param T;
param n;
param beta;

param R{1..T, 1..n};
param mean{1..n};
param minReturn;
param maxAllocation;

var x{1..n} >= 0, <= maxAllocation;
var alpha;
var u{1..T} >= 0;

minimize CVaR:
    alpha + (1 / ((1 - beta) * T)) * sum {t in 1..T} u[t];

subject to TailRisk {t in 1..T}:
    u[t] >= -sum {i in 1..n} R[t,i] * x[i] - alpha;

subject to Budget:
    sum {i in 1..n} x[i] = 1;

subject to ReturnTarget:
    sum {i in 1..n} mean[i] * x[i] >= minReturn;
"""

def solveModel(model,R, mean,
                    beta,
                    minReturn,
                    maxAllocation,
                    solver="highs"):

    T, n = R.shape
    ampl = AMPL()

    ampl.eval(model)

    ampl.param["T"] = T
    ampl.param["n"] = n

    ampl.param["beta"] = beta
    ampl.param["minReturn"] = minReturn
    ampl.param["maxAllocation"] = maxAllocation

    for t in range(T):
        for i in range(n):
            ampl.param["R"][t + 1, i + 1] = float(R[t, i])

    for i in range(n):
        ampl.param["mean"][i + 1] = float(mean[i])

    ampl.option["solver"] = solver
    ampl.solve(verbose=False)

    x = np.array([ampl.var["x"][i + 1].value() for i in range(n)])
    alpha = ampl.var["alpha"].value()
    obj = ampl.getObjective("CVaR").value()

    
    return x, alpha, obj

def solveStandardModel(R, mean,
                    beta,
                    minReturn,
                    maxAllocation,
                    solver="highs"):
    
    return solveModel(standardModel, R, mean, beta, minReturn, maxAllocation, solver)


def solveWithMoneyConstraint(R, mean,
                    beta,
                    minReturn,
                    maxAllocation,
                    moneyConstraint,
                    solver="highs"):

    extraMoneyConstraint = f"""
    subject to MinCashUsage:
    x[n] <= {moneyConstraint};
    """
    return solveModel(standardModel + extraMoneyConstraint, R, mean, beta, minReturn, maxAllocation, solver)



if __name__ == "__main__":
    import yfinance as yf

    TICKERS = ["GLD", "BND", "IGV", "VSLU", "VDC","SOCL","IHF","IYZ","XOM"]

    prices = yf.download(TICKERS, start="2023-01-01", end="2026-01-01", auto_adjust=True)["Close"]
    returns = prices.pct_change().dropna()

    R = returns.values
    mean = R.mean(axis=0)

    x, alpha, obj = solveStandardModel(
        R, mean,
        beta=0.95,
        minReturn=0.0,
        maxAllocation=1.0
    )

    print("Optimal weights:")
    for i, t in enumerate(TICKERS):
        print(f"{t}: {x[i]:.4f}")

    print("\nAlpha (VaR):", alpha)
    print("Objective (CVaR):", obj)

