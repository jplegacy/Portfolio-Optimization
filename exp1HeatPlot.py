import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# ---- Load CSV ----
file_path = "exp_results/exp1/exp1TrainingPerformance.csv"
output_file = "exp_results/exp1/exp1GrowthHeatmap.png"

df = pd.read_csv(file_path)

# ---- Clean / standardize columns ----
df.columns = ["window_size", "type", "growth"]

df["window_size"] = df["window_size"].astype(int)

# ---- Convert to percent ----
df["growth"] = df["growth"].astype(float) * 100

# ---- Pivot ----
heatmap_data = df.pivot(index="type", columns="window_size", values="growth")
heatmap_data = heatmap_data.sort_index(axis=1)

# ---- Symmetric color scaling around 0 ----
max_abs = max(abs(df["growth"].min()), abs(df["growth"].max()))

norm = colors.TwoSlopeNorm(
    vmin=-max_abs,
    vcenter=max_abs * 0.2,  # bias toward red
    vmax=max_abs
)

cmap = "RdYlGn"
# ---- Plot ----
plt.figure(figsize=(12, 4))

im = plt.imshow(heatmap_data, aspect="auto", cmap=cmap, norm=norm)

# ---- Axis labels ----
plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns, rotation=45)

# ---- Colorbar ----
cbar = plt.colorbar(im)
cbar.set_label("Growth (%)")

# ---- Add values inside each cell ----
for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        value = heatmap_data.iloc[i, j]

        if pd.notna(value):
            plt.text(
                j, i,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="black"
            )

# ---- Titles ----
plt.title("Growth Heatmap by Window Size and Type")
plt.xlabel("Window Size")
plt.ylabel("Type (daily / window)")

plt.tight_layout()

# ---- Save ----
plt.savefig(output_file, dpi=300)

plt.show()

print(f"Saved heatmap to {output_file}")