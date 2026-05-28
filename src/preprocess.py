import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import os

PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_data(path="data/raw/ai4i2020.csv"):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna()
    return df


def encode_features(df):
    le = LabelEncoder()
    df = df.copy()
    df["Type"] = le.fit_transform(df["Type"])  # H=0, L=1, M=2
    joblib.dump(le, f"{PROCESSED_DIR}/label_encoder.pkl")
    return df


def engineer_features(df):
    df = df.copy()
    df["Temp_Difference"] = df["Process temperature"] - df["Air temperature"]
    df["Power_Factor"] = df["Torque"] * df["Rotational speed"]
    return df


def split_and_scale(df):
    # Drop individual failure sub-labels; predict binary Machine failure
    feature_cols = [
        "Type", "Air temperature", "Process temperature",
        "Rotational speed", "Torque", "Tool wear",
        "Temp_Difference", "Power_Factor"
    ]
    target_col = "Machine failure"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, f"{PROCESSED_DIR}/scaler.pkl")

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, feature_cols


def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE — class distribution: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")
    return X_resampled, y_resampled


def run_preprocessing(path="data/raw/ai4i2020.csv"):
    print("Loading data...")
    df = load_data(path)
    print(f"  Raw shape: {df.shape}")

    print("Cleaning...")
    df = clean_data(df)

    print("Encoding categorical features...")
    df = encode_features(df)

    print("Engineering features...")
    df = engineer_features(df)

    print("Splitting and scaling...")
    X_train, X_test, y_train, y_test, feature_cols = split_and_scale(df)
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    print("Applying SMOTE...")
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # Save processed splits
    np.save(f"{PROCESSED_DIR}/X_train.npy", X_train_res)
    np.save(f"{PROCESSED_DIR}/X_test.npy", X_test)
    np.save(f"{PROCESSED_DIR}/y_train.npy", y_train_res)
    np.save(f"{PROCESSED_DIR}/y_test.npy", y_test)
    joblib.dump(feature_cols, f"{PROCESSED_DIR}/feature_cols.pkl")

    print("Preprocessing complete. Artifacts saved to data/processed/")
    return X_train_res, X_test, y_train_res, y_test, feature_cols


if __name__ == "__main__":
    run_preprocessing()
