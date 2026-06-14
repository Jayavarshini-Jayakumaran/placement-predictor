"""
eda.py
------
Exploratory Data Analysis for the Campus Placement Eligibility dataset.
Generates plots saved to the `plots/` folder.

Run with:
    python eda.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def save_fig(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    df = pd.read_csv("data.csv")
    print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    counts = df["eligible"].value_counts().sort_index()

    # 1. Class balance
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x="eligible", hue="eligible", data=df, palette="viridis",
                  legend=False, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Not Eligible", "Eligible"])
    ax.set_title("Class Distribution: Eligible vs Not Eligible")
    for i, v in enumerate(counts):
        ax.text(i, v + 1, str(v), ha="center", fontweight="bold")
    save_fig(fig, "01_class_balance.png")

    # 2. Feature distributions
    numeric_cols = ["cgpa", "skills", "internships", "projects"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, col in zip(axes.flatten(), numeric_cols):
        sns.histplot(df[col], kde=True, ax=ax, color="steelblue")
        ax.set_title(f"Distribution of {col}")
    fig.suptitle("Feature Distributions", y=1.02, fontsize=14)
    save_fig(fig, "02_feature_distributions.png")

    # 3. CGPA vs Eligibility
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x="eligible", y="cgpa", hue="eligible", data=df,
                palette="Set2", legend=False, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Not Eligible", "Eligible"])
    ax.set_title("CGPA vs Placement Eligibility")
    save_fig(fig, "03_cgpa_vs_eligible.png")

    # 4. Eligibility rate by skills / internships / projects
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, ["skills", "internships", "projects"]):
        rate = df.groupby(col)["eligible"].mean()
        sns.barplot(x=rate.index, y=rate.values, hue=rate.index,
                    palette="crest", legend=False, ax=ax)
        ax.set_ylabel("Eligibility Rate")
        ax.set_title(f"Eligibility Rate by {col}")
        ax.set_ylim(0, 1)
    save_fig(fig, "04_eligibility_rate_by_feature.png")

    # 5. Correlation heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    save_fig(fig, "05_correlation_heatmap.png")

    # 6. Pairplot
    pairplot = sns.pairplot(df, vars=numeric_cols, hue="eligible", palette="husl", diag_kind="kde")
    pairplot.fig.suptitle("Pairwise Relationships by Eligibility", y=1.02)
    pairplot.savefig(os.path.join(PLOTS_DIR, "06_pairplot.png"), dpi=120, bbox_inches="tight")
    plt.close(pairplot.fig)
    print(f"Saved {os.path.join(PLOTS_DIR, '06_pairplot.png')}")

    print(f"\nEDA complete. Plots saved in '{PLOTS_DIR}/'")


if __name__ == "__main__":
    main()
