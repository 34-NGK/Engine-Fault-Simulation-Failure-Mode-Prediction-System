"""
Engine Fault Simulation & Failure Mode Prediction System
--------------------------------------------------------
Author: Oluwatomi "Tomi" Oladunni

Applies predictive modeling for fault detection and classification using
Failure Mode Avoidance (FMA), Product Verification & Validation (V&V),
and System Capability Engineering (SCE) principles.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# --------------------------- Utility Functions --------------------------- #

def resolve_path(relative_path: str) -> str:
    """Return absolute path that works across systems."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(base_dir, relative_path))


def load_data(path: str, label_col: str, drop_cols=None):
    """Load dataset and prepare features/labels."""
    df = pd.read_csv(path)
    if drop_cols:
        drop_cols = [c for c in drop_cols if c in df.columns]
    else:
        drop_cols = []
    X = df.drop(columns=[label_col] + drop_cols)
    y = df[label_col]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """Split dataset into training and validation sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_model(X_train, y_train):
    """Train Random Forest model for fault classification."""
    model = RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate and return performance metrics."""
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred, average="macro"),
        "precision": precision_score(y_test, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred)
    }
    return metrics


def plot_feature_importance(model, features, output_path):
    """Generate and save feature importance chart."""
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances.values, y=importances.index, palette="crest")
    plt.title("Feature Importance - Design Validation Insight")
    plt.xlabel("Importance Weight")
    plt.ylabel("Sensor Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return importances


def generate_design_review(metrics, important_features, output_path):
    """Write a concise design review summary file."""
    summary = f"""
------------------------------------------------------------
Design Review Summary
------------------------------------------------------------
Project: Engine Fault Simulation & Failure Mode Prediction
Engineer: Oluwatomi Oladunni

Verification and Validation Results:
  - Accuracy: {metrics['accuracy']:.3f}
  - Recall: {metrics['recall']:.3f}
  - Precision: {metrics['precision']:.3f}

Top Contributing Features:
  {', '.join(important_features[:3])}

Notes:
  Model validated using balanced class weights and reproducible seed.
  Results align with Failure Mode Avoidance (FMA)
  and System Capability Engineering (SCE) principles.
------------------------------------------------------------
"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(summary)
    print(summary)


# --------------------------- Main Execution --------------------------- #

def main():
    """Execute the model training and validation workflow."""
    # Auto-detect correct data path
    data_path = resolve_path("../data/dataset_cleaned.csv")
    label_col = "Fault_Condition"
    drop_cols = ["Time_Stamp", "Operational_Mode"]

    # Load data
    X, y = load_data(data_path, label_col, drop_cols)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train and evaluate
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    # Prepare directories
    results_dir = resolve_path("../results")
    reports_dir = resolve_path("../reports")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # Save model and visuals
    joblib.dump(model, os.path.join(results_dir, "engine_fault_model.pkl"))
    importances = plot_feature_importance(
        model, X.columns, os.path.join(results_dir, "feature_importance_chart.png")
    )

    # Write report
    generate_design_review(
        metrics, importances.index.tolist(), os.path.join(reports_dir, "DesignReview_Summary.md")
    )

    print("Model training, validation, and documentation completed successfully.")


if __name__ == "__main__":
    main()
