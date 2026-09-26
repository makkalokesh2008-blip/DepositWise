from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
UNSUPERVISED_FEATURES = ["age", "balance", "day", "campaign", "pdays", "previous"]

_TRAIN_DF = None
_TEST_DF = None
_DATA_SUMMARY_CACHE = None
_EDA_STATS_CACHE = None

COLUMN_DESCRIPTIONS = {
    "age": ("Age of customer in years", "Numerical"),
    "job": ("Type of job / profession", "Categorical (Nominal)"),
    "marital": ("Marital status (married, single, divorced)", "Categorical (Nominal)"),
    "education": ("Education level attained", "Categorical (Ordinal)"),
    "default": ("Has credit in default? (yes / no)", "Binary"),
    "balance": ("Average yearly balance in rupees (₹) / currency units", "Numerical (Financial)"),
    "housing": ("Has housing loan? (yes / no)", "Binary"),
    "loan": ("Has personal loan? (yes / no)", "Binary"),
    "contact": ("Contact communication type (cellular, telephone, unknown)", "Categorical"),
    "day": ("Last contact day of the month (1 - 31)", "Numerical (Temporal)"),
    "month": ("Last contact month of year", "Categorical (Temporal)"),
    "duration": ("Last contact duration in seconds (EXCLUDED from models to prevent leakage)", "Numerical (Leakage)"),
    "campaign": ("Number of contacts performed during this campaign for this client", "Numerical"),
    "pdays": ("Number of days that passed after client was last contacted (-1 = not contacted)", "Numerical"),
    "previous": ("Number of contacts performed before this campaign for this client", "Numerical"),
    "poutcome": ("Outcome of the previous marketing campaign", "Categorical"),
    "y": ("Has the client subscribed a term deposit? (TARGET)", "Binary Target")
}


def load_train_test():
    """Load and cache training and testing datasets."""
    global _TRAIN_DF, _TEST_DF
    if _TRAIN_DF is None or _TEST_DF is None:
        _TRAIN_DF = pd.read_csv(ROOT / "data" / "train.csv")
        _TEST_DF = pd.read_csv(ROOT / "data" / "test.csv")
    return _TRAIN_DF, _TEST_DF


def get_data_summary():
    """Return backward-compatible summary dictionary."""
    train, test = load_train_test()
    global _DATA_SUMMARY_CACHE
    if _DATA_SUMMARY_CACHE is not None:
        return _DATA_SUMMARY_CACHE

    train_mem = train.memory_usage(deep=True).sum() / (1024 * 1024)
    target_counts = train["y"].value_counts().to_dict()
    total_train = len(train)

    schema = []
    for col in train.columns:
        desc, role = COLUMN_DESCRIPTIONS.get(col, ("Banking customer attribute", "Feature"))
        schema.append({
            "column": col,
            "dtype": str(train[col].dtype),
            "missing": int(train[col].isna().sum()),
            "non_null": int(train[col].notna().sum()),
            "unique": int(train[col].nunique()),
            "sample": str(train[col].iloc[0]),
            "description": desc,
            "role": role
        })

    _DATA_SUMMARY_CACHE = {
        "train_shape": train.shape,
        "test_shape": test.shape,
        "total_records": len(train),
        "total_columns": int(train.shape[1]),
        "total_cells": int(train.shape[0] * train.shape[1]),
        "memory_mb": round(float(train_mem), 2),
        "columns": train.columns.tolist(),
        "train_missing": int(train.isna().sum().sum()),
        "test_missing": int(test.isna().sum().sum()),
        "missing_by_column": {col: int(val) for col, val in train.isna().sum().items()},
        "train_duplicates": int(train.duplicated().sum()),
        "test_duplicates": int(test.duplicated().sum()),
        "train_target": target_counts,
        "target_dist": {
            "no": int(target_counts.get("no", 0)),
            "yes": int(target_counts.get("yes", 0)),
            "no_pct": round(target_counts.get("no", 0) / total_train * 100, 1),
            "yes_pct": round(target_counts.get("yes", 0) / total_train * 100, 1),
        },
        "dtypes": {column: str(train[column].dtype) for column in train.columns},
        "preview": train.head(10).to_dict("records"),
        "schema": schema,
        "sources": [
            {"filename": "train.csv", "rows": len(train), "size": "3.7 MB", "purpose": "Model Training & Stratified Cross-Validation"},
            {"filename": "test.csv", "rows": len(test), "size": "0.37 MB", "purpose": "Reference file; not used for evaluation because its rows overlap train.csv"}
        ]
    }
    return _DATA_SUMMARY_CACHE


def get_paginated_preview(page=1, per_page=20):
    """Return paginated rows from the training dataset for interactive table view."""
    train, _ = load_train_test()
    total_records = len(train)
    total_pages = max(1, int(np.ceil(total_records / per_page)))
    
    page = max(1, min(page, total_pages))
    start_idx = (page - 1) * per_page
    end_idx = min(start_idx + per_page, total_records)
    
    subset = train.iloc[start_idx:end_idx]
    
    return {
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
        "total_records": total_records,
        "start_idx": start_idx + 1,
        "end_idx": end_idx,
        "columns": train.columns.tolist(),
        "rows": subset.to_dict("records")
    }


def get_eda_stats():
    """Compute detailed descriptive and IQR outlier statistics for the EDA page."""
    global _EDA_STATS_CACHE
    if _EDA_STATS_CACHE is not None:
        return _EDA_STATS_CACHE

    train, _ = load_train_test()
    num_cols = ["age", "balance", "day", "campaign", "pdays", "previous"]
    cat_cols = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]

    # Numerical statistics
    numerical_stats = []
    for col in num_cols:
        series = train[col]
        numerical_stats.append({
            "feature": col,
            "count": int(series.count()),
            "mean": round(float(series.mean()), 2),
            "median": round(float(series.median()), 2),
            "std": round(float(series.std()), 2),
            "min": round(float(series.min()), 2),
            "p25": round(float(series.quantile(0.25)), 2),
            "p75": round(float(series.quantile(0.75)), 2),
            "max": round(float(series.max()), 2)
        })

    # Categorical statistics
    categorical_stats = []
    for col in cat_cols:
        series = train[col]
        top_val = str(series.mode()[0])
        top_count = int((series == top_val).sum())
        categorical_stats.append({
            "feature": col,
            "unique": int(series.nunique()),
            "top": top_val,
            "top_count": top_count,
            "top_pct": round(top_count / len(series) * 100, 1),
            "missing": int(series.isna().sum())
        })

    # Outlier Analysis (IQR Method)
    outlier_analysis = []
    total_rows = len(train)
    for col in num_cols:
        series = train[col]
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]
        outlier_count = int(len(outliers))
        outlier_pct = round(outlier_count / total_rows * 100, 2)

        # Domain explanation
        if col == "balance":
            rationale = "High savings balances (> ₹3,462) represent affluent high-propensity depositors; negative balances reflect overdrafts. Both are legitimate banking realities."
        elif col == "campaign":
            rationale = "Repeated contacts (> 6 calls) represent aggressive marketing attempts; valid outreach history, retained without artificial truncation."
        elif col == "pdays":
            rationale = "Value -1 denotes client was not previously contacted; values > 0 are genuine days since last interaction."
        elif col == "previous":
            rationale = "Previous contact count > 0 reflects customer history from past marketing campaigns; valid customer relationship data."
        elif col == "age":
            rationale = "Ages > 70 represent senior retired savers with high deposit propensity; valid demographic segment."
        else:
            rationale = "Natural statistical distribution; valid calendar day records."

        outlier_analysis.append({
            "feature": col,
            "q1": round(q1, 2),
            "q3": round(q3, 2),
            "iqr": round(iqr, 2),
            "lower_bound": round(lower, 2),
            "upper_bound": round(upper, 2),
            "outlier_count": outlier_count,
            "outlier_pct": outlier_pct,
            "rationale": rationale
        })

    _EDA_STATS_CACHE = {
        "numerical": numerical_stats,
        "categorical": categorical_stats,
        "outliers": outlier_analysis,
        "total_records": total_rows
    }
    return _EDA_STATS_CACHE


def get_unsupervised_sample(sample_size=1500, random_state=42):
    """Return standardized numerical banking features for unsupervised analyses.

    The target ``y`` and call duration are intentionally absent.  Each clustering
    or PCA page therefore studies customer feature structure, not subscription
    outcomes or post-call information.
    """
    train, _ = load_train_test()
    sample = train[UNSUPERVISED_FEATURES].sample(
        n=min(sample_size, len(train)), random_state=random_state
    ).copy()
    scaled = StandardScaler().fit_transform(sample)
    projection = PCA(n_components=2, random_state=random_state).fit_transform(scaled)
    return sample, scaled, projection
