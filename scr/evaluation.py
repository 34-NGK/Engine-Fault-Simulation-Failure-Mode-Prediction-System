"""
Engine Fault Evaluation and System Capability Analysis
------------------------------------------------------
Author: Oluwatomi "Tomi" Oladunni

Performs post-model evaluation and system-level validation to assess
sensor interactions, operating conditions, and feature correlations.
Supports System Capability Engineering (SCE) and Product Validation Review.
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


def resolve_path(relative_path: str) -> str:
    """Return absolute path across systems."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(base_dir, relative_path))


def load_data_and_model():
    """Load cleaned dataset and trained model."""
    data_path = resolve_path("../data/dataset_cleaned.csv")
    model_path = resolve_path("../results/engine_fault_model.pkl")

    df = pd.read_csv(data_path)
    model = joblib.load(model_path)
    return df, model


def correlation_matrix(df, output_path):
    """Generate and save correlation matrix heatmap."""
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("System Capability Correlation Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def scatter_plots(df, output_dir):
    """Generate key operational scatter plots."""
    os.makedirs(output_dir, exist_ok=True)

    if {"Temperature_(°C)", "Torque"} <= set(df.columns):
        sns.scatterplot(data=df, x="Temperature_(°C)", y="Torque", hue="Fault_Condition", palette="tab10")
        plt.title("Temperature vs Torque by Fault Condition")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "temp_vs_torque.png"), dpi=300)
        plt.close()

    if {"Rpm", "Vibration_X"} <= set(df.columns):
        sns.scatterplot(data=df, x="Rpm", y="Vibration_X", hue="Fault_Condition", palette="tab10")
        plt.title("RPM vs Vibration_X by Fault Condition")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rpm_vs_vibration.png"), dpi=300)
        plt.close()


def summarize_system_behavior(df):
    """Print summary statistics relevant to system capability."""
    print("System Capability Summary:")
    print(df.groupby("Fault_Condition")[["Temperature_(°C)", "Rpm", "Torque"]].mean())
    print("\nOperational Mode Distribution:")
    print(df["Operational_Mode"].value_counts())


def main():
    """Run full evaluation pipeline."""
    df, model = load_data_and_model()

    results_dir = resolve_path("../results/evaluation")
    os.makedirs(results_dir, exist_ok=True)

    correlation_matrix(df, os.path.join(results_dir, "correlation_matrix.png"))
    scatter_plots(df, results_dir)
    summarize_system_behavior(df)

    print("System capability evaluation and visualization completed.")


if __name__ == "__main__":
    main()
