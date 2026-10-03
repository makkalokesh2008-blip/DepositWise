"""Density-based DBSCAN clustering for DepositWise customer features."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from load_data import get_unsupervised_sample

ROOT = Path(__file__).resolve().parent
CHARTS = ROOT / "static" / "charts"
_CACHE = {}


def _line_distance(point, start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = np.hypot(dx, dy)
    return 0.0 if length == 0 else abs(dy * point[0] - dx * point[1] + end[0] * start[1] - end[1] * start[0]) / length


def run_dbscan(eps_method="auto", eps_value=1.0, min_samples=5):
    """Fit DBSCAN and identify core, border, and noise records without using y."""
    cache_key = (eps_method, round(float(eps_value), 4), int(min_samples))
    if cache_key in _CACHE:
        return _CACHE[cache_key]
    CHARTS.mkdir(parents=True, exist_ok=True)
    frame, scaled, projection = get_unsupervised_sample(sample_size=1500)
    neighbors = NearestNeighbors(n_neighbors=int(min_samples)).fit(scaled)
    distances, _ = neighbors.kneighbors(scaled)
    k_distances = np.sort(distances[:, -1])
    start, end = (0, k_distances[0]), (len(k_distances) - 1, k_distances[-1])
    knee_index = int(np.argmax([_line_distance((index, value), start, end) for index, value in enumerate(k_distances)]))
    knee_eps = float(k_distances[knee_index])
    epsilon = knee_eps if eps_method == "auto" else float(eps_value)
    model = DBSCAN(eps=epsilon, min_samples=int(min_samples))
    labels = model.fit_predict(scaled)
    core = np.zeros(len(labels), dtype=bool); core[model.core_sample_indices_] = True
    noise = labels == -1; border = ~core & ~noise; clusters = sorted(set(labels) - {-1})
    if len(clusters) >= 2:
        silhouette, sil_note = round(float(silhouette_score(scaled[~noise], labels[~noise], sample_size=min(800, (~noise).sum()), random_state=42)), 4), "Cluster separation"
    else:
        silhouette, sil_note = "N/A", "Requires two or more clusters"
    plt.figure(figsize=(7, 4.4), dpi=140); plt.plot(k_distances, color="#0ea5e9", label=f"{min_samples}-nearest-neighbor distance")
    plt.axhline(epsilon, color="#ef4444", linestyle="--", label=f"ε = {epsilon:.4f}"); plt.scatter(knee_index, knee_eps, color="#ef4444", s=70, label="Detected knee")
    plt.xlabel("Sorted customer records"); plt.ylabel("Neighbor distance"); plt.title("DBSCAN K-Distance Knee Analysis"); plt.legend(); plt.tight_layout(); plt.savefig(CHARTS / "dbscan_kdistance.png"); plt.close()
    plt.figure(figsize=(7, 4.4), dpi=140)
    if noise.any(): plt.scatter(projection[noise, 0], projection[noise, 1], marker="x", color="#94a3b8", s=22, label="Noise")
    for cluster in clusters:
        color = plt.cm.tab10(cluster % 10); cmask = labels == cluster
        plt.scatter(projection[cmask & core, 0], projection[cmask & core, 1], s=25, alpha=.72, color=color, label=f"Cluster {cluster + 1} core")
        if (cmask & border).any(): plt.scatter(projection[cmask & border, 0], projection[cmask & border, 1], s=30, alpha=.9, marker="^", color=color, label=f"Cluster {cluster + 1} border")
    plt.xlabel("Principal component 1"); plt.ylabel("Principal component 2"); plt.title("DBSCAN Customer Density Structure"); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(CHARTS / "dbscan_clusters.png"); plt.close()
    profiles = []
    for label, title, type_label in [(-1, "Noise (-1)", "Low-density records")] + [(cluster, f"Cluster {cluster + 1}", "Dense customer region") for cluster in clusters]:
        mask = labels == label
        if not mask.any(): continue
        rows = frame.iloc[np.flatnonzero(mask)]
        profiles.append({"cluster_label": title, "type": type_label, "count": int(mask.sum()), "percentage": round(float(mask.mean() * 100), 1), "avg_age": round(float(rows.age.mean()), 1), "avg_balance": round(float(rows.balance.mean()), 2), "avg_day": round(float(rows.day.mean()), 1), "avg_campaign": round(float(rows.campaign.mean()), 2), "avg_pdays": round(float(rows.pdays.mean()), 1), "avg_previous": round(float(rows.previous.mean()), 2)})
    result = {"eps_method": eps_method, "epsilon": round(float(epsilon), 4), "min_samples": int(min_samples), "clusters": len(clusters), "core_points": int(core.sum()), "border_points": int(border.sum()), "noise_points": int(noise.sum()), "silhouette": silhouette, "sil_note": sil_note, "profiles": profiles, "kdist_chart": "dbscan_kdistance.png", "scatter_chart": "dbscan_clusters.png"}
    _CACHE[cache_key] = result
    return result
