import requests

BASE = "http://127.0.0.1:5000"

routes = [
    ("/", 200),
    ("/data-loading", 200),
    ("/eda", 200),
    ("/preprocessing", 200),
    ("/linear-regression", 200),
    ("/linear-regression?view=without", 200),
    ("/linear-regression?view=regularized", 200),
    ("/linear-regression?view=compare", 200),
    ("/logistic-regression", 200),
    ("/logistic-regression?view=without", 200),
    ("/logistic-regression?view=regularized", 200),
    ("/logistic-regression?view=compare", 200),
    ("/tree-based-algorithms?model=decision_tree", 200),
    ("/tree-based-algorithms?model=random_forest", 200),
    ("/tree-based-algorithms?model=extra_trees", 200),
    ("/tree-based-algorithms?model=gradient_boosting", 200),
    ("/tree-based-algorithms?model=adaboost", 200),
    ("/tree-based-algorithms?model=xgboost", 200),
    ("/tree-based-algorithms?model=lightgbm", 200),
    ("/k-means?method=manual&k=3", 200),
    ("/k-means?method=elbow", 200),
    ("/k-means?method=silhouette", 200),
    ("/hierarchical?linkage=ward&k_selection=manual&k=3", 200),
    ("/hierarchical?linkage=complete&k_selection=auto", 200),
    ("/dbscan-clustering?eps_method=auto&min_samples=5", 200),
    ("/dbscan-clustering?eps_method=manual&eps_value=1.5&min_samples=10", 200),
    ("/pca", 200),
    ("/health", 200),
    ("/predict", 404),
]

all_ok = True
for path, expected in routes:
    try:
        resp = requests.get(BASE + path, timeout=30)
        ok = resp.status_code == expected
        tag = "OK  " if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"{tag} {resp.status_code} (expected {expected})  {path}")
    except Exception as e:
        print(f"ERR  ---  {path}  ({e})")
        all_ok = False

print()
print("=" * 50)
print("ALL ROUTES OK" if all_ok else "SOME ROUTES FAILED")
