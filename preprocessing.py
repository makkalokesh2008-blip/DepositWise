"""Reusable, leakage-safe preprocessing helpers for Deposit01 models."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
TARGET = "y"
LEAKAGE_FEATURE = "duration"
NUMERICAL_COLUMNS = ["age", "balance", "day", "campaign", "pdays", "previous"]
CATEGORICAL_COLUMNS = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]


def load_training_data():
    return pd.read_csv(ROOT / "data" / "train.csv")


def classification_data(data=None):
    data = load_training_data() if data is None else data
    x = data.drop(columns=[TARGET, LEAKAGE_FEATURE])
    y = data[TARGET].map({"no": 0, "yes": 1})
    return x, y


def classification_split(data=None):
    x, y = classification_data(data)
    return train_test_split(x, y, test_size=0.20, random_state=42, stratify=y)


def preprocessor(scale_numerical=True):
    numerical_transformer = StandardScaler() if scale_numerical else "passthrough"
    return ColumnTransformer([
        ("numerical", numerical_transformer, NUMERICAL_COLUMNS),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
    ])


def preprocessing_report(data=None):
    data = load_training_data() if data is None else data
    x_train, x_test, y_train, y_test = classification_split(data)
    standard = StandardScaler().fit(x_train[NUMERICAL_COLUMNS])
    minimum = MinMaxScaler().fit(x_train[NUMERICAL_COLUMNS])
    comparison = []
    for index, column in enumerate(NUMERICAL_COLUMNS):
        comparison.append({
            "feature": column,
            "original_min": float(x_train[column].min()), "original_max": float(x_train[column].max()),
            "standard_mean": round(float(standard.mean_[index]), 3), "standard_scale": round(float(standard.scale_[index]), 3),
            "minmax_min": round(float(minimum.data_min_[index]), 3), "minmax_max": round(float(minimum.data_max_[index]), 3),
        })
    iqr_results = []
    for column in ["age", "balance", "campaign", "pdays", "previous"]:
        q1, q3 = x_train[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        iqr_results.append({"feature": column, "q1": round(float(q1), 2), "q3": round(float(q3), 2),
                            "iqr": round(float(iqr), 2), "lower_fence": round(float(lower), 2),
                            "upper_fence": round(float(upper), 2),
                            "outlier_count": int(((x_train[column] < lower) | (x_train[column] > upper)).sum()),
                            "test_outside_fences": int(((x_test[column] < lower) | (x_test[column] > upper)).sum()),
                            "treatment": "Flagged for review; values retained (no automatic deletion or clipping)."})
    return {
        "shape": [int(data.shape[0]), int(data.shape[1])], "missing_values": int(data.isna().sum().sum()),
        "missing_by_column": {column: int(value) for column, value in data.isna().sum().items()},
        "duplicate_rows": int(data.duplicated().sum()), "target_encoding": {"no": 0, "yes": 1},
        "categorical_columns": CATEGORICAL_COLUMNS, "numerical_columns": NUMERICAL_COLUMNS,
        "duration_excluded": True, "iqr_results": iqr_results, "scaling_comparison": comparison,
        "split": {"training_percent": 80, "testing_percent": 20, "stratified": True, "random_state": 42,
                  "training_rows": int(len(x_train)), "testing_rows": int(len(x_test)),
                  "training_class_distribution": {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
                  "testing_class_distribution": {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()}},
    }
