"""Agglomerative Hierarchical Clustering for DepositWise customer features."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from load_data import get_unsupervised_sample

ROOT = Path(__file__).resolve().parent
CHARTS = ROOT / "static" / "charts"
_CACHE = {}


def _profiles(frame, labels):
    profiles = []
    for cluster in sorted(np.unique(labels)):
        rows = frame.iloc[np.flatnonzero(labels == cluster)]
        profiles.append({"cluster_id": int(cluster + 1), "count": int(len(rows)), "percentage": round(float(len(rows) / len(frame) * 100), 1), "avg_age": round(float(rows.age.mean()), 1), "avg_balance": round(float(rows.balance.mean()), 2), "avg_campaign": round(float(rows.campaign.mean()), 2), "avg_pdays": round(float(rows.pdays.mean()), 1), "avg_previous": round(float(rows.previous.mean()), 2)})
    return profiles


def run_hierarchical(linkage_method="ward", k_selection="manual", k=3):
    """Create a dendrogram and agglomerative customer segmentation."""
    cache_key = (linkage_method, k_selection, k)
    if cache_key in _CACHE:
        return _CACHE[cache_key]
    CHARTS.mkdir(parents=True, exist_ok=True)
    frame, scaled, projection = get_unsupervised_sample(sample_size=1000)
    matrix = linkage(scaled, method=linkage_method, metric="euclidean")
    if k_selection == "auto":
        candidates = range(2, 7)
        scores = []
        for candidate in candidates:
            labels = AgglomerativeClustering(n_clusters=candidate, linkage=linkage_method, metric="euclidean").fit_predict(scaled)
            # Evaluate the full 1,000-record analysis sample. Randomly subsampling
            # can omit a tiny valid cluster and make silhouette undefined.
            scores.append(silhouette_score(scaled, labels))
        selected_k = list(candidates)[int(np.argmax(scores))]
    else:
        selected_k = int(k)
    labels = AgglomerativeClustering(n_clusters=selected_k, linkage=linkage_method, metric="euclidean").fit_predict(scaled)
    cut_height = float((matrix[-(selected_k - 1), 2] + matrix[-selected_k, 2]) / 2)
    figure, axis = plt.subplots(figsize=(7, 4.4), dpi=140)
    dendrogram(matrix, truncate_mode="lastp", p=24, leaf_rotation=90, leaf_font_size=8, show_contracted=True, ax=axis, color_threshold=cut_height)
    axis.axhline(cut_height, color="#ef4444", linestyle="--", label=f"K={selected_k} cut")
    axis.set_title(f"Hierarchical Dendrogram ({linkage_method.title()} linkage)"); axis.set_xlabel("Customer groups"); axis.set_ylabel("Distance"); axis.legend(); figure.tight_layout(); figure.savefig(CHARTS / "hierarchical_dendrogram.png"); plt.close(figure)
    plt.figure(figsize=(7, 4.4), dpi=140)
    for cluster in range(selected_k):
        mask = labels == cluster; plt.scatter(projection[mask, 0], projection[mask, 1], s=20, alpha=.68, label=f"Cluster {cluster + 1}")
    plt.xlabel("Principal component 1"); plt.ylabel("Principal component 2"); plt.title(f"Hierarchical Customer Clusters (K={selected_k})"); plt.legend(); plt.tight_layout(); plt.savefig(CHARTS / "hierarchical_clusters.png"); plt.close()
    result = {"k": selected_k, "linkage": linkage_method, "k_selection": k_selection, "cut_height": round(cut_height, 2), "silhouette": round(float(silhouette_score(scaled, labels)), 4), "samples": len(frame), "profiles": _profiles(frame, labels), "dendrogram_chart": "hierarchical_dendrogram.png", "scatter_chart": "hierarchical_clusters.png"}
    _CACHE[cache_key] = result
    return result
