"""
Engine Fault Simulation & Failure Mode Prediction System
Author: Oluwatomi "Tomi" Oladunni

Applies predictive modeling for fault detection and classification using
Failure Mode Avoidance (FMA), Product Verification & Validation (V&V),
and System Capability Engineering (SCE) principles.
"""


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score
)
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# Utility: Resolve file paths to absolute locations for cross-system use
def resolve_path(relative_path: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(base_dir, relative_path))


# Load dataset and prepare feature and label sets
def load_data(path: str, label_col: str, drop_cols=None):
    df = pd.read_csv(path)

    if drop_cols:
        drop_cols = [c for c in drop_cols if c in df.columns]
    else:
        drop_cols = []

    X = df.drop(columns=[label_col] + drop_cols)
    y = df[label_col]

    print("Class distribution (original):")
    print(y.value_counts(normalize=True))
    return X, y


# Apply StandardScaler normalization to all numeric features
def preprocess_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

# Enhance features by adding key interaction and ratio terms
def feature_engineering(df):
    df = df.copy()
    if "Torque" in df.columns and "Power_Output_(Kw)" in df.columns:
        df["Torque_to_Power"] = df["Torque"] / (df["Power_Output_(Kw)"] + 1e-6)
    if "Rpm" in df.columns and "Temperature_(°C)" in df.columns:
        df["Temp_Rpm_Product"] = df["Rpm"] * df["Temperature_(°C)"]
    if "Vibration_X" in df.columns and "Vibration_Y" in df.columns and "Vibration_Z" in df.columns:
        df["Vibration_Magnitude"] = np.sqrt(df["Vibration_X"]**2 + df["Vibration_Y"]**2 + df["Vibration_Z"]**2)
    return df



# Train a tuned Random Forest model for failure mode prediction
def train_and_evaluate(X_train, X_test, y_train, y_test):
    # Feature selection for top predictors
    selector = SelectKBest(score_func=f_classif, k=min(8, X_train.shape[1]))
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)

    # Tuned XGBoost model
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softmax",
        num_class=len(np.unique(y_train)),
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_sel, y_train)
    y_pred = model.predict(X_test_sel)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred, average="macro"),
        "precision": precision_score(y_test, y_pred, average="macro"),
        "report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    # Return both model and feature selector for later use
    return model, metrics, selector.get_feature_names_out()



# Perform 5-fold cross-validation to confirm system-level reliability
def cross_validate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"Cross-validation mean accuracy: {scores.mean():.3f}")
    return scores.mean()


# Visualize and store feature importance for design review documentation
def plot_feature_importance(model, features, output_path):
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances.values, y=importances.index, palette="crest")
    plt.title("Feature Importance - Fault Classification")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return importances

def plot_feature_importance(model, features, output_path):
    """Visualize feature importances based on selected features only."""
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances.values, y=importances.index, palette="crest")
    plt.title("Feature Importance - Fault Classification (Selected Features)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return importances



# Export summarized technical documentation for Design Review
def generate_design_review(metrics, important_features, output_path):
    summary = f"""
------------------------------------------------------------
Design Review Summary
------------------------------------------------------------
Project: Engine Fault Simulation & Failure Mode Prediction
Engineer: Oluwatomi Oladunni

Verification & Validation Results:
  - Accuracy: {metrics['accuracy']:.3f}
  - Recall: {metrics['recall']:.3f}
  - Precision: {metrics['precision']:.3f}

Top Contributing Features:
  {', '.join(important_features[:3])}
Confusion Matrix:
{metrics['confusion_matrix']}   
Classification Report:
{metrics['report']}
------------------------------------------------------------
"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(summary)
    print(summary)


# Main pipeline: load data, balance, train, evaluate, and document results
def main():
    data_path = resolve_path("../data/dataset_cleaned.csv")
    label_col = "Fault_Condition"
    drop_cols = ["Time_Stamp", "Operational_Mode"]

    # Load and prepare dataset
    X, y = load_data(data_path, label_col, drop_cols)
    
    # Feature engineering before split
    X = feature_engineering(X)

    # Split data into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply SMOTE to handle class imbalance
    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
    print("After SMOTE balancing:")
    print(pd.Series(y_train_bal).value_counts())

    # Standardize features for uniform scaling
    X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train_bal, X_test)

    # Train tuned model and evaluate metrics
    model, metrics, selected_features = train_and_evaluate(X_train_scaled, X_test_scaled, y_train_bal, y_test)

    # Estimate generalization performance via 5-fold cross-validation
    cv_mean = cross_validate_model(model, X_train_scaled, y_train_bal)

    # Create output directories
    results_dir = resolve_path("../results")
    reports_dir = resolve_path("../reports")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # Save model and scaler for reuse
    joblib.dump(model, os.path.join(results_dir, "engine_fault_model.pkl"))
    joblib.dump(scaler, os.path.join(results_dir, "scaler.pkl"))

    # Plot and store feature importance chart
    importances = plot_feature_importance(
        model, selected_features, os.path.join(results_dir, "feature_importance_chart.png")
)

    

    # Generate formal design review summary report
    generate_design_review(
        metrics, importances.index.tolist(), os.path.join(reports_dir, "DesignReview_Summary.md")
    )

    print("Final Accuracy:", metrics["accuracy"])
    print("Cross-Validation Average:", round(cv_mean, 3))
    print("Training and validation completed successfully.")
    


# Entry point for script execution
if __name__ == "__main__":
    main()
