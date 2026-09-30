"""Decision-tree and ensemble classifiers for DepositWise."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, RandomForestClassifier)
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

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


def _display_feature_name(feature):
    return feature.replace("numerical__", "").replace("categorical__", "").replace("_", " ")


def train_decision_trees(data=None):
    """Fit all tree/ensemble classifiers on the same leakage-safe split."""
    CHARTS.mkdir(parents=True, exist_ok=True)
    data = load_training_data() if data is None else data
    x_train, x_test, y_train, y_test = classification_split(data)
    specifications = {
        "decision_tree": ("Decision Tree", DecisionTreeClassifier(max_depth=7, min_samples_leaf=20, random_state=42)),
        "random_forest": ("Random Forest", RandomForestClassifier(n_estimators=180, max_depth=12, min_samples_leaf=4, oob_score=True, n_jobs=-1, random_state=42)),
        "extra_trees": ("Extra Trees", ExtraTreesClassifier(n_estimators=180, max_depth=12, min_samples_leaf=4, n_jobs=-1, random_state=42)),
        "gradient_boosting": ("Gradient Boosting", GradientBoostingClassifier(n_estimators=120, max_depth=2, learning_rate=0.08, random_state=42)),
        "adaboost": ("AdaBoost", AdaBoostClassifier(n_estimators=120, learning_rate=0.5, random_state=42)),
        "xgboost": ("XGBoost", XGBClassifier(n_estimators=140, max_depth=3, learning_rate=0.08, subsample=0.85, colsample_bytree=0.85, eval_metric="logloss", n_jobs=-1, random_state=42)),
        "lightgbm": ("LightGBM", LGBMClassifier(n_estimators=140, num_leaves=24, learning_rate=0.08, subsample=0.85, colsample_bytree=0.85, n_jobs=-1, random_state=42, verbosity=-1)),
    }
    report = {"train_rows": len(x_train), "testing_rows": len(x_test), "models": {}}
    for key, (name, estimator) in specifications.items():
        pipeline = Pipeline([("preprocessing", preprocessor(scale_numerical=False)), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        probability = pipeline.predict_proba(x_test)[:, 1]
        record = {"name": name, **_metrics(y_test, pipeline.predict(x_test), probability)}
        if key == "random_forest":
            record["oob_score"] = float(pipeline.named_steps["model"].oob_score_)
        names = pipeline.named_steps["preprocessing"].get_feature_names_out()
        importance = pipeline.named_steps["model"].feature_importances_
        top = pd.DataFrame({"feature": [_display_feature_name(item) for item in names], "importance": importance}).nlargest(12, "importance").sort_values("importance")
        record["top_features"] = top.iloc[::-1].to_dict("records")
        report["models"][key] = record

        figure, axis = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(np.asarray(record["confusion_matrix"]), display_labels=["No", "Yes"]).plot(ax=axis, colorbar=False, cmap="Blues")
        axis.set_title(name); figure.tight_layout(); figure.savefig(CHARTS / f"{key}_confusion_matrix.png", dpi=150); plt.close(figure)
        plt.figure(figsize=(8, 5)); plt.barh(top["feature"], top["importance"], color="#0ea5e9")
        plt.title(f"Top {name} Feature Importances"); plt.tight_layout(); plt.savefig(CHARTS / f"{key}_importance.png", dpi=150); plt.close()
    return report
