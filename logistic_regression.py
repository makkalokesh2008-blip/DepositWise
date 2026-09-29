"""Leakage-safe Logistic Regression experiments for term-deposit subscription."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline

from preprocessing import classification_split, load_training_data, preprocessor

ROOT = Path(__file__).resolve().parent
CHARTS = ROOT / "static" / "charts"


def _metrics(actual, predicted, probability):
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1_score": float(f1_score(actual, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, probability)),
        "pr_auc": float(average_precision_score(actual, probability)),
        "confusion_matrix": [[int(value) for value in row] for row in confusion_matrix(actual, predicted)],
    }


def train_logistic_regression(data=None):
    """Train the project's two Logistic Regression specifications on one split."""
    CHARTS.mkdir(parents=True, exist_ok=True)
    data = load_training_data() if data is None else data
    x_train, x_test, y_train, y_test = classification_split(data)
    specifications = {
        "logistic_no_regularization": ("Without Regularization", LogisticRegression(C=float("inf"), max_iter=2000, random_state=42)),
        "logistic_l2": ("With Regularization", LogisticRegression(C=1.0, max_iter=2000, random_state=42)),
    }
    report = {"train_rows": len(x_train), "testing_rows": len(x_test), "models": {}}
    fitted = {}
    for key, (name, model) in specifications.items():
        pipeline = Pipeline([("preprocessing", preprocessor(scale_numerical=True)), ("model", model)])
        pipeline.fit(x_train, y_train)
        probability = pipeline.predict_proba(x_test)[:, 1]
        record = {"name": name, **_metrics(y_test, pipeline.predict(x_test), probability)}
        report["models"][key] = record
        fitted[key] = (pipeline, probability)

        fpr, tpr, _ = roc_curve(y_test, probability)
        plt.figure(figsize=(7, 5)); plt.plot(fpr, tpr, label=f"AUC = {record['roc_auc']:.3f}"); plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title(f"{name}: ROC Curve"); plt.legend(); plt.tight_layout()
        plt.savefig(CHARTS / f"{key}_roc_curve.png", dpi=150); plt.close()
        precision, recall, _ = precision_recall_curve(y_test, probability)
        plt.figure(figsize=(7, 5)); plt.plot(recall, precision, label=f"PR-AUC = {record['pr_auc']:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"{name}: Precision-Recall Curve"); plt.legend(); plt.tight_layout()
        plt.savefig(CHARTS / f"{key}_pr_curve.png", dpi=150); plt.close()
        figure, axis = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(np.asarray(record["confusion_matrix"]), display_labels=["No", "Yes"]).plot(ax=axis, colorbar=False, cmap="Blues")
        axis.set_title(name); figure.tight_layout(); figure.savefig(CHARTS / f"{key}_confusion_matrix.png", dpi=150); plt.close(figure)

    names = fitted["logistic_l2"][0].named_steps["preprocessing"].get_feature_names_out()
    coefficients = fitted["logistic_l2"][0].named_steps["model"].coef_[0]
    coefficient_table = pd.DataFrame({"feature": names, "coefficient": coefficients})
    report["models"]["logistic_l2"]["strongest_positive"] = coefficient_table.nlargest(10, "coefficient").to_dict("records")
    report["models"]["logistic_l2"]["strongest_negative"] = coefficient_table.nsmallest(10, "coefficient").to_dict("records")
    _comparison_charts(report, fitted, y_test)
    return report


def _comparison_charts(report, fitted, actual):
    labels = ["Without Regularization", "With Regularization"]
    keys = ["logistic_no_regularization", "logistic_l2"]
    x = np.arange(5)
    metric_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    plt.figure(figsize=(8, 5))
    for offset, key, label in [(-0.18, keys[0], labels[0]), (0.18, keys[1], labels[1])]:
        plt.bar(x + offset, [report["models"][key][metric] for metric in metric_names], 0.36, label=label)
    plt.xticks(x, ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]); plt.ylim(0, 1); plt.title("Logistic Regression Metric Comparison"); plt.legend(); plt.tight_layout()
    plt.savefig(CHARTS / "logistic_metric_comparison.png", dpi=150); plt.close()
    plt.figure(figsize=(7, 5))
    for key, label in zip(keys, labels):
        fpr, tpr, _ = roc_curve(actual, fitted[key][1]); plt.plot(fpr, tpr, label=label)
    plt.plot([0, 1], [0, 1], "--", color="gray"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("Logistic Regression ROC Comparison"); plt.legend(); plt.tight_layout()
    plt.savefig(CHARTS / "logistic_roc_comparison.png", dpi=150); plt.close()
    plt.figure(figsize=(7, 5))
    for key, label in zip(keys, labels):
        precision, recall, _ = precision_recall_curve(actual, fitted[key][1]); plt.plot(recall, precision, label=label)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Logistic Regression Precision-Recall Comparison"); plt.legend(); plt.tight_layout()
    plt.savefig(CHARTS / "logistic_pr_comparison.png", dpi=150); plt.close()
    figure, axes = plt.subplots(1, 2, figsize=(8, 4))
    for axis, key, label in zip(axes, keys, labels):
        ConfusionMatrixDisplay(np.asarray(report["models"][key]["confusion_matrix"]), display_labels=["No", "Yes"]).plot(ax=axis, colorbar=False, cmap="Blues"); axis.set_title(label)
    figure.tight_layout(); figure.savefig(CHARTS / "logistic_confusion_comparison.png", dpi=150); plt.close(figure)
