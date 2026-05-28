import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    matthews_corrcoef, RocCurveDisplay
)

MODELS_DIR = "models"
PROCESSED_DIR = "data/processed"
FIGURES_DIR = "results/figures"
METRICS_DIR = "results/metrics"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

MODEL_FILES = {
    "Random Forest": "random_forest.pkl",
    "SVM": "svm.pkl",
    "Logistic Regression": "logistic_regression.pkl",
}


def load_test_data():
    X_test = np.load(f"{PROCESSED_DIR}/X_test.npy")
    y_test = np.load(f"{PROCESSED_DIR}/y_test.npy")
    return X_test, y_test


def compute_metrics(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }, y_pred, y_prob


def plot_confusion_matrix(name, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Failure", "Failure"],
                yticklabels=["No Failure", "Failure"], ax=ax)
    ax.set_title(f"Confusion Matrix — {name}")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    fname = name.lower().replace(" ", "_")
    fig.savefig(f"{FIGURES_DIR}/cm_{fname}.png", dpi=150)
    plt.close(fig)


def plot_roc_curves(models_data, X_test, y_test):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in models_data.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", label="Random Chance")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/roc_curves.png", dpi=150)
    plt.close(fig)
    print("  Saved: results/figures/roc_curves.png")


def plot_feature_importance(feature_cols):
    rf = joblib.load(f"{MODELS_DIR}/random_forest.pkl")
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(feature_cols)), importances[indices])
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels([feature_cols[i] for i in indices], rotation=45, ha="right")
    ax.set_title("Random Forest — Feature Importances")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/feature_importance.png", dpi=150)
    plt.close(fig)
    print("  Saved: results/figures/feature_importance.png")


def run_evaluation():
    print("Loading test data...")
    X_test, y_test = load_test_data()

    models = {}
    for name, fname in MODEL_FILES.items():
        path = f"{MODELS_DIR}/{fname}"
        if not os.path.exists(path):
            print(f"  Skipping {name} — model file not found.")
            continue
        models[name] = joblib.load(path)

    if not models:
        print("No trained models found. Run train.py first.")
        return

    all_metrics = []
    print("\nEvaluating models...")
    for name, model in models.items():
        metrics, y_pred, y_prob = compute_metrics(name, model, X_test, y_test)
        all_metrics.append(metrics)
        plot_confusion_matrix(name, y_test, y_pred)
        print(f"  {name}: F1={metrics['F1-Score']}  AUC={metrics['ROC-AUC']}  MCC={metrics['MCC']}")

    results_df = pd.DataFrame(all_metrics).set_index("Model")
    results_df.to_csv(f"{METRICS_DIR}/summary.csv")
    print(f"\nMetrics saved to results/metrics/summary.csv")
    print(results_df.to_string())

    print("\nPlotting ROC curves...")
    plot_roc_curves(models, X_test, y_test)

    feature_cols = joblib.load(f"{PROCESSED_DIR}/feature_cols.pkl")
    print("Plotting feature importances...")
    plot_feature_importance(feature_cols)

    return results_df


if __name__ == "__main__":
    run_evaluation()
