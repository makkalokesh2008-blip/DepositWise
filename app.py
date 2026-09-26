"""Flask routes for the DepositWise banking analytics dashboard."""

import json
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

from DBSCAN import run_dbscan
from decision_trees import train_decision_trees
from deposit01_eda import generate_eda
from hierarchical_clustering import run_hierarchical
from kmeans import run_kmeans
from linear_regression import train_linear_regression
from load_data import get_data_summary, get_eda_stats, get_paginated_preview
from logistic_regression import train_logistic_regression
from pca import run_pca
from preprocessing import load_training_data, preprocessing_report

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)


def _write_report(name: str, report: dict[str, Any]) -> None:
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / name).write_text(json.dumps(report, indent=2), encoding="utf-8")


def artifacts() -> dict[str, Any]:
    """Load current reports, rebuilding only when topic-based reports are absent."""
    paths = {
        "classification": ROOT / "reports" / "classification_report.json",
        "regression": ROOT / "reports" / "regression_report.json",
        "preprocessing": ROOT / "reports" / "preprocessing_report.json",
    }
    needs_build = not all(path.exists() for path in paths.values())
    if not needs_build:
        existing = json.loads(paths["classification"].read_text(encoding="utf-8"))
        needs_build = existing.get("report_version") != "topic-based-v9"
    if needs_build:
        data = load_training_data()
        logistic = train_logistic_regression(data)
        trees = train_decision_trees(data)
        _write_report("classification_report.json", {
            "report_version": "topic-based-v9",
            "train_rows": logistic["train_rows"],
            "testing_rows": logistic["testing_rows"],
            "models": {**logistic["models"], **trees["models"]},
        })
        _write_report("regression_report.json", train_linear_regression(data))
        _write_report("preprocessing_report.json", preprocessing_report(data))
    return {key: json.loads(path.read_text(encoding="utf-8")) for key, path in paths.items()}


ARTIFACTS = artifacts()


@app.route("/")
def home():
    return render_template("index.html", active="home", summary=get_data_summary())


@app.route("/data-loading")
def data_loading():
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    if per_page not in [10, 20, 50, 100]:
        per_page = 20
    return render_template("data_loading.html", active="data-loading", summary=get_data_summary(), preview=get_paginated_preview(page=page, per_page=per_page))


@app.route("/eda")
def eda():
    chart = ROOT / "static" / "charts" / "target_distribution.png"
    if not chart.exists():
        generate_eda()
    return render_template("eda.html", active="eda", eda_stats=get_eda_stats())


@app.route("/preprocessing")
def preprocessing():
    return render_template("preprocessing.html", active="preprocessing", report=ARTIFACTS["preprocessing"])


@app.route("/linear-regression")
def linear_regression():
    selected = request.args.get("view", "compare")
    if selected not in ["without", "regularized", "compare"]:
        selected = "compare"
    return render_template("linear_regression.html", active="linear-regression", report=ARTIFACTS["regression"], selected=selected)


@app.route("/logistic-regression")
def logistic_regression():
    selected = request.args.get("view", "compare")
    if selected not in ["without", "regularized", "compare"]:
        selected = "compare"
    model_key = "logistic_no_regularization" if selected == "without" else "logistic_l2"
    return render_template("logistic_regression.html", active="logistic-regression", report=ARTIFACTS["classification"], selected=selected, model_key=model_key)


@app.route("/tree-based-algorithms")
def tree_based_algorithms():
    allowed = ["decision_tree", "random_forest", "extra_trees", "gradient_boosting", "adaboost", "xgboost", "lightgbm"]
    selected = request.args.get("model", "random_forest")
    return render_template("tree_algorithms.html", active="tree-based-algorithms", report=ARTIFACTS["classification"], selected=selected if selected in allowed else "random_forest")


@app.route("/k-means")
def k_means():
    method = request.args.get("method", "manual")
    k = request.args.get("k", 3, type=int)
    return render_template("k_means.html", active="k-means", result=run_kmeans(k=k if 2 <= k <= 8 else 3, method=method if method in ["manual", "elbow", "silhouette"] else "manual"))


@app.route("/hierarchical")
def hierarchical_clustering():
    linkage = request.args.get("linkage", "ward")
    selection = request.args.get("k_selection", "manual")
    k = request.args.get("k", 3, type=int)
    return render_template("hierarchical.html", active="hierarchical", result=run_hierarchical(linkage_method=linkage if linkage in ["ward", "complete", "average", "single"] else "ward", k_selection=selection if selection in ["manual", "auto"] else "manual", k=k if 2 <= k <= 6 else 3))


@app.route("/dbscan-clustering")
def dbscan_clustering():
    method = request.args.get("eps_method", "auto")
    epsilon = request.args.get("eps_value", 1.0, type=float)
    min_samples = request.args.get("min_samples", 5, type=int)
    return render_template("dbscan.html", active="dbscan", result=run_dbscan(eps_method=method if method in ["auto", "manual"] else "auto", eps_value=epsilon if 0.1 <= epsilon <= 10.0 else 1.0, min_samples=min_samples if 2 <= min_samples <= 100 else 5))


@app.route("/pca")
def pca_analysis():
    return render_template("pca.html", active="pca", result=run_pca())


@app.route("/health")
def health():
    return jsonify(status="ok")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
