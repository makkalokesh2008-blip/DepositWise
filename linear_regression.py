"""Secondary continuous-value regression experiment for DepositWise.

The primary project target is binary subscription ``y``.  This module therefore
uses ``balance`` only as a separate continuous target and deliberately excludes
balance itself, y, and duration from its predictors.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import NUMERICAL_COLUMNS, load_training_data, preprocessor

ROOT = Path(__file__).resolve().parent
CHARTS = ROOT / "static" / "charts"


def _metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "mse": float(mse),
        "rmse": float(mse ** 0.5),
        "r2": float(r2_score(actual, predicted)),
    }


def train_linear_regression(data=None):
    """Fit OLS and Ridge on the same train/test split and create their charts."""
    CHARTS.mkdir(parents=True, exist_ok=True)
    data = load_training_data() if data is None else data
    features = data.drop(columns=["balance", "y", "duration"])
    target = data["balance"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, random_state=42
    )
    numerical = [column for column in NUMERICAL_COLUMNS if column != "balance"]
    outcomes = {}
    predictions = {}
    specifications = {
        "linear": ("Linear Regression (OLS)", LinearRegression()),
        "ridge": ("Ridge Regression (L2)", Ridge(alpha=1.0)),
    }

    for key, (name, estimator) in specifications.items():
        transformer = preprocessor(scale_numerical=True)
        transformer.transformers[0] = ("numerical", transformer.transformers[0][1], numerical)
        pipeline = Pipeline([("preprocessing", transformer), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        prediction = pipeline.predict(x_test)
        outcomes[key] = {"name": name, **_metrics(y_test, prediction)}
        predictions[key] = prediction

        plt.figure(figsize=(7, 4.4), dpi=140)
        plt.scatter(y_test.iloc[:600], prediction[:600], alpha=0.5, color="#0ea5e9", s=18)
        low = min(y_test.iloc[:600].min(), prediction[:600].min())
        high = max(y_test.iloc[:600].max(), prediction[:600].max())
        plt.plot([low, high], [low, high], "--", color="#ef4444", label="Ideal fit (y = x)")
        plt.xlabel("Actual balance (₹)")
        plt.ylabel("Predicted balance (₹)")
        plt.title(f"{name}: Actual vs Predicted Balance")
        plt.legend()
        plt.tight_layout()
        plt.savefig(CHARTS / f"regression_{key}_actual_predicted.png")
        plt.close()

    plt.figure(figsize=(7, 4.4), dpi=140)
    labels = [outcomes[key]["name"] for key in specifications]
    values = [outcomes[key]["r2"] for key in specifications]
    bars = plt.bar(labels, values, color=["#0284c7", "#0ea5e9"], width=0.48)
    plt.ylabel("R² score")
    plt.title("Regression Comparison: Explained Variance")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.001, f"{value:.4f}", ha="center")
    plt.tight_layout()
    plt.savefig(CHARTS / "regression_comparison.png")
    plt.close()

    return {"target": "balance", "train_rows": len(x_train), "testing_rows": len(x_test), "models": outcomes}
