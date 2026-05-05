import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import matplotlib.dates as mdates

import os

from amplpy import AMPL
from tqdm import tqdm
from model import standardModel
import json

from tools import run_with_money_constraint, total_return


# =========================================================
# CONFIG
# =========================================================


# Benchmark ETFs and 10 holdings
pairUps = {
 "SPY":["META","GOOGL","GOOG","NFLX","VZ","T","DIS","TMUS","WBD","CMCSA",
        "WMT","COST","PG","KO","PEP","PM","MO","MDLZ","CL","TGT",
        "XOM","CVX","COP","WMB","EOG","MPC","VLO","PSX","SLB","KMI",
        "JPM","BRK-B","MA","V","BAC","WFC","GS","MS","C","HSBC",
        "LLY","JNJ","ABBV","MRK","UNH","AMGN","TMO","ABT","GILD","ISRG",
        "CAT","GE","RTX","GEV","BA","DE","HON","UBER","ETN","LMT",
        "NVDA","AAPL","MSFT","AVGO","MU","AMD","PLTR","CSCO","AMAT","LRCX",
        "LIN","NEM","FCX","SHW","CRH","ECL","APD","CTVA","NUE","VMC",
        "NEE","SO","DUK","AEP","SRE","D","VST","ETR","XEL","CEG",
        "WELL","PLD","EQIX","AMT","DLR","SPG","O","PSA","CBRE","ICE"]
 }

# Comment to 
BACKING = "GLD"
for etf, holdings in pairUps.items():
    holdings.append(BACKING)

TICKERS = [ticker for etf, holdings in pairUps.items() for ticker in holdings]
BENCHMARKS = [etf for etf in pairUps.keys()]


START_DATE = "2025-01-01"
END_DATE = "2026-01-01"

BETA = 0.95
MIN_RETURN = 0.0
MAX_ALLOCATION = 1.0
INFLATION_RATE = 0.03
MONEY_CONSTRAINT = 0

SOLVER = "highs"

WINDOWS = [14]
FREQS = ["window"]

SHOW_PLOT = False

RESULT_DIR = "exp_results/exp3"
PLOT_PATH = os.path.join(RESULT_DIR, "exp3ConglomoratePlot.png")
RESULTS_CSV = os.path.join(RESULT_DIR, "exp3ConglomoratePerformance.csv")

os.makedirs(RESULT_DIR, exist_ok=True)

ETF_SECTORS = {
    "SPY": "Overall Market",
}
    
# =========================================================
# MAIN
# =========================================================

def main():
    # ---------------- DATA LOAD ----------------
    all_tickers = list(set([t for holdings in pairUps.values() for t in holdings]))
    
    prices = yf.download(
        all_tickers,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True
    )["Close"]

    returns = prices.pct_change().dropna()

    etf_prices = yf.download(
        BENCHMARKS,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True
    )["Close"]

    etf_returns = etf_prices.pct_change().dropna()

    # inflation-adjusted buying power
    buying_power = (1 - INFLATION_RATE)**(1/365) - 1

    # ---------------- PLOTTING SETUP ----------------
    fig, ax = plt.subplots(figsize=(20, 8))

    results_summary = []

    # ---------------- LOOP OVER ETF MATCHUPS ----------------
    for i, (etf, tickers) in enumerate(pairUps.items()):
        print(f"\nRunning test for {etf}...")

        sub_returns = returns[tickers].copy()
        sub_returns["BUYING_POWER"] = buying_power

        strat_returns, _ = run_with_money_constraint(
            sub_returns,
            WINDOWS[0],
            FREQS[0],
            MONEY_CONSTRAINT,
            BETA,
            MIN_RETURN,
            MAX_ALLOCATION,
            solver=SOLVER
        )

        strat_cum = (1 + strat_returns).cumprod() - 1
        etf_cum = (1 + etf_returns[etf].loc[strat_cum.index]).cumprod() - 1

        strat_total = total_return(strat_returns)
        etf_total = total_return(etf_returns[etf].loc[strat_returns.index])

        results_summary.append({
            "ETF": etf,
            "Strategy": strat_total,
            "ETF Return": etf_total,
            "Excess": strat_total - etf_total
        })

        # ---------------- PLOT (SAME STYLE AS BEFORE) ----------------
        ax.plot(strat_cum, label="Strategy" if i == 0 else None, color="blue",linewidth=3.0)
        ax.plot(etf_cum, label="ETF" if i == 0 else None, color="red",linewidth=3.0)

    # ---------------- FINAL FORMATTING (MATCH YOUR OLD STYLE) ----------------
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2)

    ax.set_title(
        "SPY Universe Strategy vs ETF Performance",
        fontsize=10
    )

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha('right')

    ax.grid(True)

    ax.legend()

    # ---------------- FINALIZE ----------------
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()

    if SHOW_PLOT:
        plt.show()
    else:
        plt.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
        plt.close()
    # ---------------- PRINT SUMMARY ----------------
    summary_df = pd.DataFrame(results_summary)
    print("\n===== FINAL RESULTS =====")
    print(summary_df.sort_values("Excess", ascending=False))

    summary_df.to_csv(RESULTS_CSV, index=False)

    
# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()