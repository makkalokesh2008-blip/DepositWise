"""K-Means customer segmentation using DepositWise numerical banking features."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from load_data import get_unsupervised_sample

ROOT = Path(__file__).resolve().parent
CHARTS = ROOT / "static" / "charts"
_CACHE = {}


def _line_distance(point, start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = np.hypot(dx, dy)
    return 0.0 if length == 0 else abs(dy * point[0] - dx * point[1] + end[0] * start[1] - end[1] * start[0]) / length


def _profiles(frame, labels):
    profiles = []
    for cluster in sorted(np.unique(labels)):
        rows = frame.iloc[np.flatnonzero(labels == cluster)]
        profiles.append({
            "cluster_id": int(cluster + 1), "count": int(len(rows)), "percentage": round(len(rows) / len(frame) * 100, 1),
            "avg_age": round(float(rows.age.mean()), 1), "avg_balance": round(float(rows.balance.mean()), 2),
            "avg_day": round(float(rows.day.mean()), 1), "avg_campaign": round(float(rows.campaign.mean()), 2),
            "avg_pdays": round(float(rows.pdays.mean()), 1), "avg_previous": round(float(rows.previous.mean()), 2),
        })
    return profiles


def run_kmeans(k=3, method="manual"):
    """Run manual, elbow-selected, or silhouette-selected K-Means."""
    cache_key = (k, method)
    if cache_key in _CACHE:
        return _CACHE[cache_key]
    CHARTS.mkdir(parents=True, exist_ok=True)
    frame, scaled, projection = get_unsupervised_sample(sample_size=1500)
    candidates = list(range(2, 9))
    fits = [KMeans(n_clusters=value, n_init=10, random_state=42).fit(scaled) for value in candidates]
    inertias = [float(fit.inertia_) for fit in fits]
    silhouettes = [float(silhouette_score(scaled, fit.labels_, sample_size=800, random_state=42)) for fit in fits]
    if method == "elbow":
        start, end = (candidates[0], inertias[0]), (candidates[-1], inertias[-1])
        selected_k = candidates[int(np.argmax([_line_distance((value, inertias[index]), start, end) for index, value in enumerate(candidates)]))]
    elif method == "silhouette":
        selected_k = candidates[int(np.argmax(silhouettes))]
    else:
        selected_k = int(k)
    model = KMeans(n_clusters=selected_k, n_init=10, random_state=42)
    labels = model.fit_predict(scaled)

    plt.figure(figsize=(7, 4.4), dpi=140)
    if method == "silhouette":
        plt.plot(candidates, silhouettes, "o-", color="#0ea5e9", label="Silhouette score"); plt.ylabel("Mean silhouette score")
    else:
        plt.plot(candidates, inertias, "o-", color="#0284c7", label="Inertia (WCSS)"); plt.ylabel("Inertia (WCSS)")
    plt.axvline(selected_k, color="#ef4444", linestyle="--", label=f"Selected K = {selected_k}")
    plt.xlabel("Number of clusters (K)"); plt.title("K-Means Cluster Count Analysis"); plt.legend(); plt.tight_layout(); plt.savefig(CHARTS / "kmeans_analysis.png"); plt.close()
    plt.figure(figsize=(7, 4.4), dpi=140)
    for cluster in range(selected_k):
        mask = labels == cluster; plt.scatter(projection[mask, 0], projection[mask, 1], s=20, alpha=.68, label=f"Cluster {cluster + 1}")
    centers = PCA(n_components=2, random_state=42).fit(scaled).transform(model.cluster_centers_)
    plt.scatter(centers[:, 0], centers[:, 1], marker="X", color="#0f172a", s=110, label="Centroids")
    plt.xlabel("Principal component 1"); plt.ylabel("Principal component 2"); plt.title(f"K-Means Customer Segments (K={selected_k})"); plt.legend(); plt.tight_layout(); plt.savefig(CHARTS / "kmeans_clusters.png"); plt.close()
    result = {"k": selected_k, "method": method, "inertia": round(float(model.inertia_), 2), "silhouette": round(float(silhouette_score(scaled, labels, sample_size=800, random_state=42)), 4), "samples": len(frame), "profiles": _profiles(frame, labels), "analysis_chart": "kmeans_analysis.png", "scatter_chart": "kmeans_clusters.png"}
    _CACHE[cache_key] = result
    return result
