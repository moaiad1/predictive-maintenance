"""
Generate methodology figures for the predictive maintenance project report.
Outputs all figures to results/figures/methodology/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import joblib

OUT_DIR = "results/figures/methodology"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
BLUE   = "#2C6FAC"
ORANGE = "#E07B39"
GREEN  = "#3A9E6E"
PURPLE = "#7B52AB"
GREY   = "#6B6B6B"
LIGHT  = "#F0F4FA"


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 — Pipeline Flowchart
# ──────────────────────────────────────────────────────────────────────────────
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    steps = [
        ("1  Data Collection",        "AI4I 2020 dataset\n10,000 records · 12 columns",          BLUE),
        ("2  Data Cleaning",           "Remove duplicates & nulls\nDrop irrelevant sub-labels",    BLUE),
        ("3  Feature Engineering",     "Temp Difference = Process T − Air T\nPower Factor = Torque × RPM", GREEN),
        ("4  Encoding & Scaling",      "LabelEncoder (machine type)\nRobustScaler (numeric feats)",        PURPLE),
        ("5  Train / Test Split",      "80 % train · 20 % test\nStratified by class",               PURPLE),
        ("6  Class Balancing (SMOTE)", "Synthetic Minority Over-sampling\nBalances minority failure class", ORANGE),
        ("7  Model Training",          "Random Forest · SVM (RBF)\nLogistic Regression + GridSearchCV",   BLUE),
        ("8  Evaluation",              "Accuracy · Precision · Recall\nF1 · ROC-AUC · MCC",         GREEN),
    ]

    box_w, box_h = 6.5, 0.72
    x0 = 1.75
    gap = 0.28
    y_start = 9.3

    for i, (title, detail, color) in enumerate(steps):
        y = y_start - i * (box_h + gap)

        # Shadow
        shadow = FancyBboxPatch((x0 + 0.05, y - box_h - 0.05), box_w, box_h,
                                boxstyle="round,pad=0.04", linewidth=0,
                                facecolor="#cccccc", zorder=1)
        ax.add_patch(shadow)

        # Main box
        box = FancyBboxPatch((x0, y - box_h), box_w, box_h,
                             boxstyle="round,pad=0.04", linewidth=1.2,
                             edgecolor=color, facecolor=LIGHT, zorder=2)
        ax.add_patch(box)

        # Coloured left bar
        bar = FancyBboxPatch((x0, y - box_h), 0.22, box_h,
                             boxstyle="round,pad=0.0", linewidth=0,
                             facecolor=color, zorder=3)
        ax.add_patch(bar)

        ax.text(x0 + 0.38, y - box_h * 0.35, title,
                fontsize=10, fontweight="bold", color="#1a1a1a", va="center", zorder=4)
        ax.text(x0 + 0.38, y - box_h * 0.72, detail,
                fontsize=7.8, color=GREY, va="center", zorder=4)

        # Arrow between boxes
        if i < len(steps) - 1:
            arrow_y = y - box_h - 0.01
            ax.annotate("", xy=(x0 + box_w / 2, arrow_y - gap + 0.02),
                        xytext=(x0 + box_w / 2, arrow_y),
                        arrowprops=dict(arrowstyle="-|>", color="#444444",
                                       lw=1.4, mutation_scale=14), zorder=5)

    ax.set_title("Predictive Maintenance — Methodology Pipeline",
                 fontsize=13, fontweight="bold", pad=8)
    plt.tight_layout()
    path = f"{OUT_DIR}/fig1_pipeline.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 — Dataset Overview (class distribution + feature box-plots)
# ──────────────────────────────────────────────────────────────────────────────
def fig_dataset_overview():
    df = pd.read_csv("data/raw/ai4i2020.csv")

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # 2a — Class distribution
    ax0 = fig.add_subplot(gs[0, 0])
    counts = df["Machine failure"].value_counts().sort_index()
    bars = ax0.bar(["No Failure\n(0)", "Failure\n(1)"], counts.values,
                   color=[BLUE, ORANGE], edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, counts.values):
        ax0.text(bar.get_x() + bar.get_width() / 2, v + 60,
                 f"{v:,}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=9, fontweight="bold")
    ax0.set_title("Class Distribution", fontweight="bold")
    ax0.set_ylabel("Sample Count")
    ax0.set_ylim(0, max(counts.values) * 1.18)
    ax0.spines[["top", "right"]].set_visible(False)

    # 2b — Machine Type distribution
    ax1 = fig.add_subplot(gs[0, 1])
    type_counts = df["Type"].value_counts().sort_index()
    ax1.bar(type_counts.index, type_counts.values,
            color=[GREEN, PURPLE, BLUE], edgecolor="white")
    ax1.set_title("Machine Type Distribution", fontweight="bold")
    ax1.set_ylabel("Count")
    ax1.spines[["top", "right"]].set_visible(False)
    for i, (t, v) in enumerate(type_counts.items()):
        ax1.text(i, v + 40, str(v), ha="center", fontsize=9, fontweight="bold")

    # 2c — Failure sub-types
    ax2 = fig.add_subplot(gs[0, 2])
    sub = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    sub_counts = df[sub].sum().sort_values(ascending=False)
    colors = [ORANGE, BLUE, GREEN, PURPLE, GREY]
    ax2.barh(sub_counts.index, sub_counts.values, color=colors)
    ax2.set_title("Failure Sub-Type Counts", fontweight="bold")
    ax2.set_xlabel("Count")
    ax2.spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(sub_counts.values):
        ax2.text(v + 1, i, str(v), va="center", fontsize=9, fontweight="bold")

    # 2d-2f — Box-plots of key numeric features split by failure class
    numeric_feats = [
        ("Torque", "Torque (Nm)"),
        ("Tool wear", "Tool Wear (min)"),
        ("Rotational speed", "Rotational Speed (rpm)"),
    ]
    for idx, (col, label) in enumerate(numeric_feats):
        ax = fig.add_subplot(gs[1, idx])
        data_no  = df.loc[df["Machine failure"] == 0, col]
        data_yes = df.loc[df["Machine failure"] == 1, col]
        bp = ax.boxplot([data_no, data_yes], patch_artist=True, notch=False,
                        medianprops=dict(color="white", linewidth=2))
        bp["boxes"][0].set_facecolor(BLUE)
        bp["boxes"][1].set_facecolor(ORANGE)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["No Failure", "Failure"])
        ax.set_title(label, fontweight="bold")
        ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Dataset Overview — AI4I 2020 Predictive Maintenance",
                 fontsize=13, fontweight="bold", y=1.01)
    path = f"{OUT_DIR}/fig2_dataset_overview.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3 — Class Imbalance & SMOTE Effect
# ──────────────────────────────────────────────────────────────────────────────
def fig_smote():
    X_train_raw = np.load("data/processed/X_train.npy")  # after SMOTE
    y_train_smote = np.load("data/processed/y_train.npy")

    # Reconstruct approximate before-SMOTE distribution from raw CSV
    df = pd.read_csv("data/raw/ai4i2020.csv")
    before_0 = int((df["Machine failure"] == 0).sum() * 0.8)
    before_1 = int((df["Machine failure"] == 1).sum() * 0.8)
    after_0 = int((y_train_smote == 0).sum())
    after_1 = int((y_train_smote == 1).sum())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, (n0, n1, title, note) in zip(axes, [
        (before_0, before_1, "Before SMOTE\n(Training Set)", f"Minority: {before_1} ({before_1/(before_0+before_1)*100:.1f}%)"),
        (after_0,  after_1,  "After SMOTE\n(Resampled Training Set)",  f"Minority: {after_1} ({after_1/(after_0+after_1)*100:.1f}%)")
    ]):
        bars = ax.bar(["No Failure", "Failure"], [n0, n1],
                      color=[BLUE, ORANGE], edgecolor="white", linewidth=0.8, width=0.5)
        for bar, v in zip(bars, [n0, n1]):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 20,
                    f"{v:,}", ha="center", fontsize=10, fontweight="bold")
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_ylabel("Sample Count")
        ax.set_ylim(0, max(n0, n1) * 1.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.98, 0.97, note, transform=ax.transAxes, fontsize=9,
                ha="right", va="top", color=GREY,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=LIGHT, edgecolor=GREY, alpha=0.8))

    fig.suptitle("Class Imbalance Before vs. After SMOTE",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{OUT_DIR}/fig3_smote.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4 — Model Comparison (grouped bar chart)
# ──────────────────────────────────────────────────────────────────────────────
def fig_model_comparison():
    df = pd.read_csv("results/metrics/summary.csv", index_col="Model")
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "MCC"]
    models  = df.index.tolist()
    x       = np.arange(len(metrics))
    width   = 0.22
    colors  = [BLUE, ORANGE, GREEN]

    fig, ax = plt.subplots(figsize=(13, 5.5))

    for i, (model, color) in enumerate(zip(models, colors)):
        vals = df.loc[model, metrics].values.astype(float)
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=model,
                      color=color, edgecolor="white", linewidth=0.7)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold", color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Performance Comparison — All Evaluation Metrics",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0.5, color=GREY, linewidth=0.8, linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = f"{OUT_DIR}/fig4_model_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 5 — Feature Engineering Diagram
# ──────────────────────────────────────────────────────────────────────────────
def fig_feature_engineering():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    raw_features = [
        "Type (H/L/M)",
        "Air Temperature (K)",
        "Process Temperature (K)",
        "Rotational Speed (rpm)",
        "Torque (Nm)",
        "Tool Wear (min)",
    ]

    eng_features = [
        "Temp Difference\n(Process T − Air T)",
        "Power Factor\n(Torque × RPM)",
    ]

    final_features = raw_features + [f.split("\n")[0] for f in eng_features]

    # Raw features column
    ax.text(1.2, 4.7, "Original Features (6)", fontsize=10,
            fontweight="bold", ha="center", color=BLUE)
    for i, feat in enumerate(raw_features):
        y = 4.1 - i * 0.62
        box = FancyBboxPatch((0.1, y - 0.22), 2.2, 0.4,
                             boxstyle="round,pad=0.05", linewidth=1,
                             edgecolor=BLUE, facecolor="#EBF2FB")
        ax.add_patch(box)
        ax.text(1.2, y, feat, ha="center", va="center", fontsize=8.5, color="#1a1a1a")

    # Arrow cluster → engineering box
    ax.annotate("", xy=(4.0, 2.6), xytext=(2.35, 2.6),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.8, mutation_scale=16))

    # Engineering box
    eng_box = FancyBboxPatch((4.05, 1.5), 3.2, 2.2,
                             boxstyle="round,pad=0.08", linewidth=1.5,
                             edgecolor=GREEN, facecolor="#EBF9F2")
    ax.add_patch(eng_box)
    ax.text(5.65, 3.85, "Feature Engineering", ha="center", fontsize=10,
            fontweight="bold", color=GREEN)
    for i, feat in enumerate(eng_features):
        y = 3.3 - i * 1.0
        box = FancyBboxPatch((4.25, y - 0.32), 2.8, 0.6,
                             boxstyle="round,pad=0.05", linewidth=1,
                             edgecolor=GREEN, facecolor="white")
        ax.add_patch(box)
        ax.text(5.65, y, feat, ha="center", va="center", fontsize=8.5, color="#1a1a1a")

    # Arrow → final feature set
    ax.annotate("", xy=(8.0, 2.6), xytext=(7.3, 2.6),
                arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.8, mutation_scale=16))

    # Final feature set
    ax.text(10.0, 4.7, "Final Feature Set (8)", fontsize=10,
            fontweight="bold", ha="center", color=PURPLE)
    for i, feat in enumerate(final_features):
        y = 4.1 - i * 0.49
        color = PURPLE if i >= 6 else "#2C6FAC"
        bg    = "#F4F0FB" if i >= 6 else LIGHT
        box = FancyBboxPatch((8.1, y - 0.19), 3.8, 0.36,
                             boxstyle="round,pad=0.04", linewidth=1,
                             edgecolor=color, facecolor=bg)
        ax.add_patch(box)
        ax.text(10.0, y, feat, ha="center", va="center", fontsize=8, color="#1a1a1a")

    # Legend
    handles = [
        mpatches.Patch(facecolor=LIGHT, edgecolor=BLUE, label="Original feature"),
        mpatches.Patch(facecolor="#F4F0FB", edgecolor=PURPLE, label="Engineered feature"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.9)
    ax.set_title("Feature Engineering Overview", fontsize=13, fontweight="bold")

    plt.tight_layout()
    path = f"{OUT_DIR}/fig5_feature_engineering.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    fig_pipeline()
    fig_dataset_overview()
    fig_smote()
    fig_model_comparison()
    fig_feature_engineering()
    print("\nAll methodology figures saved to", OUT_DIR)
