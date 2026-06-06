"""Generate EDA figures: correlation heatmap, feature histograms, pairplot."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR = "results/figures/eda"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv("data/raw/ai4i2020.csv")

# encode Type for numeric analysis
df["Type_enc"] = df["Type"].map({"L": 0, "M": 1, "H": 2})
df["Temp_Difference"] = df["Process temperature"] - df["Air temperature"]
df["Power_Factor"]    = df["Torque"] * df["Rotational speed"]

numeric_cols = [
    "Air temperature", "Process temperature", "Rotational speed",
    "Torque", "Tool wear", "Temp_Difference", "Power_Factor",
]

# ── 1. Correlation heatmap ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
corr_cols = numeric_cols + ["Machine failure"]
corr = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax,
            annot_kws={"size": 8})
ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold", pad=10)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/eda_correlation_heatmap.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("Saved eda_correlation_heatmap.png")

# ── 2. Feature distribution histograms (by failure class) ─────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
colors = {"No Failure": "#2C6FAC", "Failure": "#E07B39"}

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    for label, grp in df.groupby("Machine failure"):
        name = "Failure" if label == 1 else "No Failure"
        ax.hist(grp[col], bins=35, alpha=0.6, label=name,
                color=colors[name], edgecolor="none", density=True)
    ax.set_title(col, fontweight="bold", fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("Density" if i % 4 == 0 else "")
    ax.spines[["top", "right"]].set_visible(False)
    if i == 0:
        ax.legend(fontsize=8)

# hide unused subplot
axes[-1].set_visible(False)
fig.suptitle("Feature Distributions by Failure Class", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/eda_feature_histograms.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("Saved eda_feature_histograms.png")

# ── 3. Pairplot (sampled for speed) ──────────────────────────────────────────
pair_cols = ["Torque", "Rotational speed", "Tool wear",
             "Temp_Difference", "Power_Factor", "Machine failure"]
sample = df[pair_cols].copy()
sample["Machine failure"] = sample["Machine failure"].map({0: "No Failure", 1: "Failure"})
# sample to keep pairplot manageable
sample = pd.concat([
    sample[sample["Machine failure"] == "No Failure"].sample(600, random_state=42),
    sample[sample["Machine failure"] == "Failure"],
])

g = sns.pairplot(sample, hue="Machine failure",
                 palette={"No Failure": "#2C6FAC", "Failure": "#E07B39"},
                 plot_kws={"alpha": 0.45, "s": 18},
                 diag_kind="kde", corner=True)
g.figure.suptitle("Pairplot of Key Features (coloured by failure class)",
                   y=1.01, fontsize=12, fontweight="bold")
g.figure.savefig(f"{OUT_DIR}/eda_pairplot.png", dpi=150, bbox_inches="tight")
plt.close(g.figure)
print("Saved eda_pairplot.png")

print(f"\nAll EDA figures saved to {OUT_DIR}")
