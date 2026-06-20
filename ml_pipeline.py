"""
ml_pipeline.py
Handles everything for the Machine Learning section:
- reading CSV columns (for the target-selection dropdown)
- preprocessing (missing values, encoding)
- training 4 classifiers
- cross-validation
- feature importance chart
"""

import io
import base64

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI backend needed on a server
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from pandas.api.types import is_numeric_dtype


def fig_to_base64(fig):
    """Convert a matplotlib figure into a base64 string we can embed
    directly inside an HTML <img> tag (no need to save image files)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def get_csv_columns(csv_path):
    """Read just the header row so we can list column names in a dropdown,
    without loading the whole file into memory."""
    df = pd.read_csv(csv_path, nrows=0)
    return list(df.columns)


def run_ml_pipeline(csv_path, target_column):
    """
    Loads a CSV, preprocesses it, trains 4 classifiers with cross-validation,
    and returns a dictionary of results ready to plug into a template.
    """
    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found in dataset.")

    # Drop rows where the target itself is missing — can't learn from those
    df = df.dropna(subset=[target_column])

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # --- Handle missing values ---
    # Numeric columns: fill with median. Categorical columns: fill with mode.
    for col in X.columns:
        if is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())
        else:
            mode_vals = X[col].mode()
            fill_val = mode_vals.iloc[0] if not mode_vals.empty else "missing"
            X[col] = X[col].fillna(fill_val)

    # --- Encode categorical feature columns into numbers ---
    for col in X.columns:
        if not is_numeric_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # --- Encode target if it's text (e.g. "pass"/"fail") ---
    if not is_numeric_dtype(y):
        target_encoder = LabelEncoder()
        y = pd.Series(target_encoder.fit_transform(y.astype(str)), index=y.index)

    # --- Train/test split ---
    can_stratify = y.nunique() > 1 and y.value_counts().min() >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if can_stratify else None
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(),
    }

    results = []
    best_model_name = None
    best_accuracy = -1
    feature_importance_chart = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # 5-fold cross-validation (or fewer folds if the dataset is small)
        cv_folds = min(5, y.value_counts().min()) if can_stratify else min(5, len(y))
        cv_folds = max(cv_folds, 2)
        cv_scores = cross_val_score(model, X, y, cv=cv_folds)

        results.append({
            "name": name,
            "test_accuracy": round(acc * 100, 2),
            "cv_mean": round(cv_scores.mean() * 100, 2),
            "cv_std": round(cv_scores.std() * 100, 2),
        })

        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name

        # Feature importance is only meaningful for tree-based models
        if name == "Random Forest":
            importances = model.feature_importances_
            order = np.argsort(importances)[::-1]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(range(len(importances)), importances[order])
            ax.set_xticks(range(len(importances)))
            ax.set_xticklabels(X.columns[order], rotation=45, ha="right")
            ax.set_ylabel("Importance")
            ax.set_title("Feature Importance (Random Forest)")
            feature_importance_chart = fig_to_base64(fig)

    return {
        "results": results,
        "best_model": best_model_name,
        "best_accuracy": round(best_accuracy * 100, 2),
        "n_rows": len(df),
        "n_features": X.shape[1],
        "target_column": target_column,
        "feature_importance_chart": feature_importance_chart,
    }



