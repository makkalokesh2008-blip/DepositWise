"""Generate the EDA charts displayed by the Deposit01 Flask application."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from load_data import load_train_test

ROOT = Path(__file__).resolve().parent
CHARTS = ROOT / "static" / "charts"
CORRELATION_FEATURES = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
NUMERICAL_TARGET_PLOTS = ["age", "balance", "campaign", "pdays"]
CATEGORICAL_TARGET_PLOTS = ["job", "education", "housing", "poutcome"]


def save_figure(name):
    plt.tight_layout()
    plt.savefig(CHARTS / name, dpi=140, facecolor="white")
    plt.close()


def generate_eda():
    train, test = load_train_test()
    CHARTS.mkdir(parents=True, exist_ok=True)

    counts = train["y"].value_counts().reindex(["no", "yes"])
    plt.figure(figsize=(6, 4)); plt.bar(counts.index, counts.values, color=["#456b91", "#18bfff"])
    plt.title("Term Deposit Subscription Distribution"); plt.xlabel("Subscription"); plt.ylabel("Customers")
    save_figure("target_distribution.png")

    numeric_for_corr = train[CORRELATION_FEATURES].copy(); numeric_for_corr["subscription_yes"] = train["y"].eq("yes").astype(int)
    correlation = numeric_for_corr.corr()
    plt.figure(figsize=(9, 7)); image = plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(image, label="Correlation"); plt.xticks(range(len(correlation)), correlation.columns, rotation=45, ha="right"); plt.yticks(range(len(correlation)), correlation.columns); plt.title("Numerical Feature Correlation Matrix")
    save_figure("correlation_matrix.png")

    for column in NUMERICAL_TARGET_PLOTS:
        grouped = [train.loc[train["y"] == label, column] for label in ["no", "yes"]]
        plt.figure(figsize=(6, 4)); plt.boxplot(grouped, tick_labels=["No", "Yes"], showfliers=False)
        plt.title(f"{column.title()} by Subscription"); plt.xlabel("Subscribed to term deposit"); plt.ylabel(column.title())
        save_figure(f"{column}_vs_target.png")

    for column in CATEGORICAL_TARGET_PLOTS:
        subscription_rate = train.groupby(column, observed=True)["y"].apply(lambda values: values.eq("yes").mean()).sort_values(ascending=False)
        plt.figure(figsize=(7, 4)); plt.bar(subscription_rate.index.astype(str), subscription_rate.values * 100, color="#18bfff")
        plt.title(f"Subscription Rate by {column.title()}"); plt.xlabel(column.title()); plt.ylabel("Subscribed (%)"); plt.xticks(rotation=35, ha="right")
        save_figure(f"{column}_vs_subscription.png")

    print(f"EDA charts regenerated from {len(train):,} training rows; test rows available: {len(test):,}.")


if __name__ == "__main__":
    generate_eda()
