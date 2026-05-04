import pandas as pd
import matplotlib.pyplot as plt

# === CHANGE THIS ===
file_path = "exp_results/exp1TestPerformance.csv"

# Load data
df = pd.read_csv(file_path)

asset_map = {
    "GLD": "Gold",
    "BND": "Bonds",
    "IGV": "Tech",
    "VDC": "Consumer(Necc)",
    "SOCL": "Media",
    "IHF": "Healthcare",
    "IYZ": "Telecom",
    "XOM": "Energy"
}

df = df.rename(columns=asset_map)

# Identify asset columns
asset_cols = [col for col in df.columns if col not in ["window", "freq", "count", "growth"]]



# Group data
grouped = list(df.groupby(["window", "freq"]))
n_groups = len(grouped)

# Create subplots in one row
fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 4), sharey=True)

# If only one group, make axes iterable
if n_groups == 1:
    axes = [axes]

for ax, ((window, freq), group) in zip(axes, grouped):
    group = group.sort_values("count")

    # shift x-axis by warm-up window
    x = group["count"] + window

    for asset in asset_cols:
        ax.plot(x, group[asset], label=asset)

    ax.set_title(f"W={window}, F={freq}")
    ax.set_xlabel("Time (Trading Days)")
    ax.grid(True)

    # optional: show warm-up region
    ax.axvspan(0, window, color="gray", alpha=0.1, label="Warm-up")
    
    ax.set_title(f"W={window}, F={freq}")
    ax.set_xlabel("Decisions")
    ax.grid(True)
    

handles, labels = axes[0].get_legend_handles_labels()

# Move legend UP (key change here)
fig.legend(
    handles, labels,
    loc="lower center",
    ncol=len(asset_cols)+1,
    bbox_to_anchor=(0.5, 0.02)   # <-- was negative, now slightly above bottom
)

fig.suptitle("Asset Allocations by (Window, Frequency)")

# Reserve just enough space
plt.tight_layout(rect=[0, 0.08, 1, 1])


plt.savefig(f"exp_results/exp1AssetPlot.png")

plt.show()