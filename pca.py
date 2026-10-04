"""Principal Component Analysis for standardized DepositWise numerical features."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from load_data import UNSUPERVISED_FEATURES, get_unsupervised_sample

ROOT = Path(__file__).resolve().parent
CHARTS = ROOT / "static" / "charts"
_CACHE = None


def run_pca():
    """Project six standardized banking features without using y or duration."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    CHARTS.mkdir(parents=True, exist_ok=True)
    _, scaled, _ = get_unsupervised_sample(sample_size=2500)
    model = PCA(n_components=len(UNSUPERVISED_FEATURES), random_state=42)
    transformed = model.fit_transform(scaled)
    explained = model.explained_variance_ratio_ * 100
    cumulative = np.cumsum(explained)
    components = np.arange(1, len(explained) + 1)
    plt.figure(figsize=(7, 4.4), dpi=140); plt.bar(components, explained, color="#0ea5e9", label="Individual variance")
    plt.plot(components, cumulative, "o--", color="#ef4444", label="Cumulative variance")
    plt.xticks(components, [f"PC{item}" for item in components]); plt.xlabel("Principal component"); plt.ylabel("Explained variance (%)"); plt.title("PCA Scree Plot"); plt.legend(); plt.tight_layout(); plt.savefig(CHARTS / "pca_scree_plot.png"); plt.close()
    plt.figure(figsize=(7, 4.4), dpi=140); plt.scatter(transformed[:, 0], transformed[:, 1], s=18, alpha=.62, color="#0ea5e9")
    plt.xlabel(f"PC1 ({explained[0]:.1f}% variance)"); plt.ylabel(f"PC2 ({explained[1]:.1f}% variance)"); plt.title("PCA: Customer Feature Space Projection"); plt.tight_layout(); plt.savefig(CHARTS / "pca_scatter.png"); plt.close()
    variance_table = [{"component": f"PC{index + 1}", "eigenvalue": round(float(model.explained_variance_[index]), 4), "explained_ratio": round(float(explained[index]), 2), "cumulative_ratio": round(float(cumulative[index]), 2)} for index in range(len(explained))]
    loadings = [{"feature": feature, **{f"PC{component + 1}": round(float(model.components_[component, index]), 4) for component in range(4)}} for index, feature in enumerate(UNSUPERVISED_FEATURES)]
    _CACHE = {"features": UNSUPERVISED_FEATURES, "explained_variance": [round(float(item), 2) for item in explained], "cumulative_variance": [round(float(item), 2) for item in cumulative], "variance_table": variance_table, "loadings": loadings, "scree_chart": "pca_scree_plot.png", "scatter_chart": "pca_scatter.png"}
    return _CACHE
