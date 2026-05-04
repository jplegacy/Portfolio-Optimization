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
 "VOX":["META","GOOGL","GOOG","NFLX","VZ","T","DIS","TMUS","WBD","CMCSA"],
 "VDC":["WMT","COST","PG","KO","PEP","PM","MO","MDLZ","CL","TGT"],
 "VDE":["XOM","CVX","COP","WMB","EOG","MPC","VLO","PSX","SLB","KMI"],
 "VFH":["JPM","BRK-B","MA","V","BAC","WFC","GS","MS","C","HSBC"],
 "VHT":["LLY","JNJ","ABBV","MRK","UNH","AMGN","TMO","ABT","GILD","ISRG"],
 "VIS":["CAT","GE","RTX","GEV","BA","DE","HON","UBER","ETN","LMT"],
 "VGT":["NVDA","AAPL","MSFT","AVGO","MU","AMD","PLTR","CSCO","AMAT","LRCX"],
 "VAW":["LIN","NEM","FCX","SHW","CRH","ECL","APD","CTVA","NUE","VMC"],
 "VPU":["NEE","SO","DUK","AEP","SRE","D","VST","ETR","XEL","CEG"],
 "VNQ":["WELL","PLD","EQIX","AMT","DLR","SPG","O","PSA","CBRE","ICE"],
 }

# Comment to 
# BACKING = "GLD"
# for etf, holdings in pairUps.items():
#     holdings.append(BACKING)
    

TICKERS = [ticker for etf, holdings in pairUps.items() for ticker in holdings]
BENCHMARKS = [etf for etf in pairUps.keys()]

START_DATE = "2025-01-01"
END_DATE = "2026-04-01"

BETA = 0.95
MIN_RETURN = 0.0
MAX_ALLOCATION = 1.0
INFLATION_RATE = 0.03
MONEY_CONSTRAINT = 0.25

SOLVER = "highs"

WINDOWS = [14]
FREQS = ["window"]

SHOW_PLOT = False

RESULT_DIR = "exp_results/exp3"
PLOT_PATH = os.path.join(RESULT_DIR, "exp3Plot.png")
RESULTS_CSV = os.path.join(RESULT_DIR, "exp3Performance.csv")

os.makedirs(RESULT_DIR, exist_ok=True)

ETF_SECTORS = {
    "VOX": "Communication Services",
    "VDC": "Consumer Staples",
    "VDE": "Energy",
    "VFH": "Financials",
    "VHT": "Healthcare",
    "VIS": "Industrials",
    "VGT": "Information Technology",
    "VAW": "Materials",
    "VPU": "Utilities",
    "VNQ": "Real Estate"
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
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    results_summary = []

    # ---------------- LOOP OVER ETF MATCHUPS ----------------
    for i, (etf, tickers) in enumerate(pairUps.items()):
        print(f"\nRunning test for {etf}...")

        # subset returns for this ETF's holdings
        sub_returns = returns[tickers].copy()
        sub_returns["BUYING_POWER"] = buying_power

        # run strategy (single configuration)
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

        # cumulative returns
        strat_cum = (1 + strat_returns).cumprod() - 1
        etf_cum = (1 + etf_returns[etf].loc[strat_cum.index]).cumprod() - 1

        # total return
        strat_total = total_return(strat_returns)
        etf_total = total_return(etf_returns[etf].loc[strat_returns.index])

        results_summary.append({
            "ETF": etf,
            "Strategy": strat_total,
            "ETF Return": etf_total,
            "Excess": strat_total - etf_total
        })

        # ---------------- PLOT ----------------
        ax = axes[i]

        # Plot lines
        ax.plot(strat_cum, label="Strategy", color="blue")
        ax.plot(etf_cum, label="ETF", color="red")

        ax.axhline(0, color="black", linestyle="--", linewidth=1.2)

        sector_name = ETF_SECTORS.get(etf, etf)

        ax.set_title(
            f"{sector_name}\nStrat: {strat_total:.2%} | {etf}: {etf_total:.2%}",
            fontsize=10
        )

        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha('right')

        ax.grid(True)

        if i == 0:
            ax.legend()

    # ---------------- FINALIZE PLOT ----------------
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