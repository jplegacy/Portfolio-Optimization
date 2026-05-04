import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import os

from amplpy import AMPL
from tqdm import tqdm
from main import ampl_model, solve_cvar_ampl
import json


# =========================================================
# CONFIG
# =========================================================

TICKERS = ["GLD", "BND", "IGV", "VSLU", "VDC","SOCL","IHF","IYZ","XOM"]
BENCHMARK = "SPY"

START_DATE = "2023-01-01"
END_DATE = "2026-01-01"

TRAIN_START = "2023-01-01"
TRAIN_END   = "2024-12-31"
TEST_START  = "2025-01-01"
TEST_END    = "2026-01-01"

BETA = 0.95
MIN_RETURN = 0.0
MAX_ALLOCATION = 1.0

SOLVER = "cplex"

WINDOWS = list(range(7, 91, 7))
FREQS = ["window","daily"]

TOP_N = 3

SHOW_PLOT = False

RESULT_DIR = "exp_results"
RESULT_CSV = os.path.join(RESULT_DIR, "exp1TrainingPerformance.csv")
TOP_WEIGHTS_CSV = os.path.join(RESULT_DIR, "exp1TopPortfolios.csv")
PLOT_PATH = os.path.join(RESULT_DIR, "exp1Plot.png")

TEST_CSV = os.path.join(RESULT_DIR, "exp1TestPerformance.csv")

os.makedirs(RESULT_DIR, exist_ok=True)

# =========================================================
# BACKTEST
# =========================================================

def run_strategy(returns, window, freq):
    n = returns.shape[1]
    weights = np.zeros(n)

    portfolio_returns = []
    weights_history = []

    for t in range(window, len(returns)):
        if freq == "daily" or (freq == "window" and (t - window) % window == 0):
            R_window = returns.iloc[t-window:t].values
            mean = R_window.mean(axis=0)

            weights, _, _ = solve_cvar_ampl(
                R_window,
                mean,
                beta=BETA,
                minReturn=MIN_RETURN,
                maxAllocation=MAX_ALLOCATION
            )

        r = returns.iloc[t].values @ weights
        portfolio_returns.append(r)
        weights_history.append(weights.copy())

    return (
        pd.Series(portfolio_returns, index=returns.index[window:]),
        np.array(weights_history)   # shape: (T, N)
    )

# =========================================================
# METRICS
# =========================================================

def total_return(r):
    if isinstance(r, pd.DataFrame):
        r = r.iloc[:, 0]
    return float((1 + r).prod() - 1)

# =========================================================
# WORKER FUNCTION (MUST BE TOP LEVEL)
# =========================================================

def evaluate_params(args):
    window, freq, train_returns = args
    try:
        strat_returns, final_weights = run_strategy(train_returns, window, freq)
        growth = total_return(strat_returns)

        return {
            "window": window,
            "freq": freq,
            "growth": growth,
            "weights": final_weights.tolist()   # JSON serializable
        }
    except Exception as e:
        print(f"Failed: window={window}, freq={freq}, error={e}")
        return None

# =========================================================
# MAIN
# =========================================================

def main():
    # initialize file
    with open(RESULT_CSV, "w") as f:
        header = "window,freq,growth," + ",".join(TICKERS) + "\n"
        f.write(header)
    # ---------------- DATA LOAD ----------------
    prices = yf.download(
        TICKERS,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True
    )["Close"]

    returns = prices.pct_change().dropna()

    spy_prices = yf.download(
        BENCHMARK,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True
    )["Close"]

    spy_returns = spy_prices.pct_change().dropna()

    # ---------------- SPLIT ----------------
    train_returns = returns.loc[TRAIN_START:TRAIN_END]
    test_returns  = returns.loc[TEST_START:TEST_END]

    spy_test = spy_returns.loc[TEST_START:TEST_END]

    # ---------------- PARALLEL SEARCH ----------------
    param_list = [(w, f, train_returns) for w in WINDOWS for f in FREQS]

    results = []

    # keep small (AMPL + CPLEX not friendly to many processes)
    with mp.Pool(mp.cpu_count()) as pool, open(RESULT_CSV, "a") as f:
        for res in tqdm(
            pool.imap_unordered(evaluate_params, param_list),
            total=len(param_list),
            desc="Hyperparameter Search"
        ):
            if res:
                results.append(res)

                weights_array = np.array(res["weights"][-1])  # last weight vector

                f.write(
                    f"{res['window']},"
                    f"{res['freq']},"
                    f"{res['growth']},"
                    + ",".join(map(str, weights_array))
                    + "\n"
                )
                f.flush()
                
    clean_results = []

    for r in results:
        if r is None:
            continue

        row = {
            "window": r["window"],
            "freq": r["freq"],
            "growth": r["growth"],
        }

        final_w = np.array(r["weights"][-1])

        for t, w in zip(TICKERS, final_w):
            row[t] = w

        clean_results.append(row)

    results_df = pd.DataFrame(clean_results)
    
    # ---------------- TOP STRATEGIES ----------------
    topN = results_df.sort_values("growth", ascending=False).head(TOP_N)
    print(f"\nTop {TOP_N} strategies (by growth):")
    print(topN)

    topN.to_csv(TOP_WEIGHTS_CSV, index=False)

    # ---------------- TEST ----------------
    plt.figure(figsize=(12, 6))

    for _, row in topN.iterrows():
        window = int(row["window"])
        freq = row["freq"]

        strat_returns, _ = run_strategy(test_returns, window, freq)
        cum_returns = strat_returns.cumsum()

        plt.plot(cum_returns, label=f"window={window}, freq={freq}")

    spy_cum = spy_test.cumsum()
    plt.plot(spy_cum, linestyle="--", linewidth=2, label="SPY", color="red")

    plt.title(f"Top {len(topN)} CVaR Strategies vs S&P 500 (2025)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid()

    if SHOW_PLOT:
        plt.show()
    else:
        plt.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
        plt.close()

    # ---------------- FINAL RESULTS ----------------
    print("\nFinal 2025 Performance:")

    spy_total = total_return(spy_test)
    print(f"SPY: {spy_total:.2%}")

    for _, row in topN.iterrows():
        window = int(row["window"])
        freq = row["freq"]
        
        strat_returns, _ = run_strategy(test_returns, window, freq)
        strat_total = total_return(strat_returns)

        print(f"Strategy (window={window}, freq={freq}): "
              f"{strat_total:.2%} | excess: {(strat_total - spy_total):.2%}")

    rows = []

    solution_counter = {}

    for _, row in topN.iterrows():
        window = int(row["window"])
        freq = row["freq"]

        key = (window, freq)
        solution_counter.setdefault(key, 0)

        strat_returns, weights_history = run_strategy(test_returns, window, freq)
        strat_total = total_return(strat_returns)

        solution_counter[key] += 1
        count = solution_counter[key]

        final_weights = weights_history[-1]

        row_dict = {
            "window": window,
            "freq": freq,
            "count": count,
            "growth": strat_total
        }

        # 👇 each asset becomes its own column
        for ticker, weight in zip(TICKERS, final_weights):
            row_dict[ticker] = weight

        rows.append(row_dict)

    df = pd.DataFrame(rows)

    # optional: enforce column order
    df = df[["window", "freq", "count", "growth"] + TICKERS]

    df.to_csv(TEST_CSV, index=False)
# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()