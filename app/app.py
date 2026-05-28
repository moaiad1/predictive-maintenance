import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    confusion_matrix, RocCurveDisplay,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef
)

# ── Paths ──────────────────────────────────────────────────────────────────────
MODELS_DIR = "models"
PROCESSED_DIR = "data/processed"
FIGURES_DIR = "results/figures"
METRICS_DIR = "results/metrics"

MODEL_FILES = {
    "Random Forest": f"{MODELS_DIR}/random_forest.pkl",
    "SVM": f"{MODELS_DIR}/svm.pkl",
    "Logistic Regression": f"{MODELS_DIR}/logistic_regression.pkl",
}

# ── Loaders ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    return {name: joblib.load(path) for name, path in MODEL_FILES.items() if os.path.exists(path)}

@st.cache_data
def load_test_data():
    X = np.load(f"{PROCESSED_DIR}/X_test.npy")
    y = np.load(f"{PROCESSED_DIR}/y_test.npy")
    return X, y

@st.cache_data
def load_metrics():
    path = f"{METRICS_DIR}/summary.csv"
    if os.path.exists(path):
        return pd.read_csv(path, index_col="Model")
    return None

@st.cache_data
def load_raw_data():
    return pd.read_csv("data/raw/ai4i2020.csv")

@st.cache_data
def load_feature_cols():
    return joblib.load(f"{PROCESSED_DIR}/feature_cols.pkl")

# ── App ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Predictive Maintenance", layout="wide", page_icon="⚙️")
st.title("⚙️ Predictive Maintenance Dashboard")
st.caption("AI4I 2020 Dataset · Random Forest · SVM · Logistic Regression")

models = load_models()
X_test, y_test = load_test_data()
feature_cols = load_feature_cols()

pages = ["Overview", "Model Comparison", "Confusion Matrices", "ROC Curves",
         "Feature Importance", "Live Prediction"]
page = st.sidebar.radio("Navigate", pages)

# ── Overview ───────────────────────────────────────────────────────────────────
if page == "Overview":
    st.header("Dataset Overview")
    df = load_raw_data()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Features", "8 (incl. engineered)")
    failures = df["Machine failure"].sum()
    col3.metric("Failures", f"{failures} ({failures/len(df)*100:.1f}%)")
    col4.metric("Models Trained", len(models))

    st.subheader("Raw Data Sample")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Class Distribution")
    fig, ax = plt.subplots(figsize=(5, 3))
    counts = df["Machine failure"].value_counts()
    ax.bar(["No Failure", "Failure"], counts.values, color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 50, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Failure Sub-Type Counts")
    sub_types = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    sub_counts = df[sub_types].sum().sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    sub_counts.plot(kind="bar", ax=ax2, color="#4C72B0")
    ax2.set_ylabel("Count")
    ax2.set_xticklabels(sub_counts.index, rotation=0)
    plt.tight_layout()
    st.pyplot(fig2)

# ── Model Comparison ───────────────────────────────────────────────────────────
elif page == "Model Comparison":
    st.header("Model Comparison")
    metrics_df = load_metrics()
    if metrics_df is not None:
        st.dataframe(metrics_df.style.highlight_max(axis=0, color="#d4f1c0"), use_container_width=True)

        st.subheader("Metrics Bar Chart")
        metric_choice = st.selectbox("Select metric", metrics_df.columns.tolist(), index=3)
        fig, ax = plt.subplots(figsize=(6, 4))
        metrics_df[metric_choice].plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452", "#55A868"])
        ax.set_ylim(0, 1)
        ax.set_ylabel(metric_choice)
        ax.set_xticklabels(metrics_df.index, rotation=15, ha="right")
        for i, v in enumerate(metrics_df[metric_choice]):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Run the pipeline first to generate metrics.")

# ── Confusion Matrices ─────────────────────────────────────────────────────────
elif page == "Confusion Matrices":
    st.header("Confusion Matrices")
    cols = st.columns(len(models))
    for col, (name, model) in zip(cols, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Failure", "Failure"],
                    yticklabels=["No Failure", "Failure"], ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        col.pyplot(fig)

# ── ROC Curves ─────────────────────────────────────────────────────────────────
elif page == "ROC Curves":
    st.header("ROC Curves")
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", label="Random Chance")
    ax.legend(loc="lower right")
    ax.set_title("ROC Curves — All Models")
    plt.tight_layout()
    st.pyplot(fig)

# ── Feature Importance ─────────────────────────────────────────────────────────
elif page == "Feature Importance":
    st.header("Random Forest — Feature Importances")
    rf = models.get("Random Forest")
    if rf:
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(range(len(feature_cols)), importances[indices], color="#4C72B0")
        ax.set_xticks(range(len(feature_cols)))
        ax.set_xticklabels([feature_cols[i] for i in indices], rotation=30, ha="right")
        ax.set_ylabel("Importance")
        ax.set_title("Feature Importances")
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Importance Table")
        importance_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": importances
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
        st.dataframe(importance_df, use_container_width=True)
    else:
        st.warning("Random Forest model not found.")

# ── Live Prediction ────────────────────────────────────────────────────────────
elif page == "Live Prediction":
    st.header("Live Failure Prediction")
    st.write("Enter sensor readings to predict failure probability.")

    scaler = joblib.load(f"{PROCESSED_DIR}/scaler.pkl")
    le = joblib.load(f"{PROCESSED_DIR}/label_encoder.pkl")

    col1, col2 = st.columns(2)
    with col1:
        machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
        air_temp = st.slider("Air Temperature (K)", 295.0, 305.0, 300.0, 0.1)
        proc_temp = st.slider("Process Temperature (K)", 305.0, 315.0, 310.0, 0.1)
        rot_speed = st.slider("Rotational Speed (rpm)", 1168, 2886, 1500)
    with col2:
        torque = st.slider("Torque (Nm)", 3.8, 76.6, 40.0, 0.1)
        tool_wear = st.slider("Tool Wear (min)", 0, 253, 100)
        model_choice = st.selectbox("Model", list(models.keys()))

    type_encoded = le.transform([machine_type])[0]
    temp_diff = proc_temp - air_temp
    power_factor = torque * rot_speed

    raw = np.array([[type_encoded, air_temp, proc_temp, rot_speed,
                     torque, tool_wear, temp_diff, power_factor]])
    scaled = scaler.transform(raw)

    if st.button("Predict", type="primary"):
        model = models[model_choice]
        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1]

        if pred == 1:
            st.error(f"⚠️ FAILURE PREDICTED — Probability: {prob:.1%}")
        else:
            st.success(f"✅ No Failure Predicted — Failure Probability: {prob:.1%}")

        st.metric("Failure Probability", f"{prob:.1%}")
        st.progress(float(prob))
