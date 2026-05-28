import numpy as np
import joblib
import os
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

MODELS_DIR = "models"
PROCESSED_DIR = "data/processed"
os.makedirs(MODELS_DIR, exist_ok=True)


def load_processed_data():
    X_train = np.load(f"{PROCESSED_DIR}/X_train.npy")
    X_test = np.load(f"{PROCESSED_DIR}/X_test.npy")
    y_train = np.load(f"{PROCESSED_DIR}/y_train.npy")
    y_test = np.load(f"{PROCESSED_DIR}/y_test.npy")
    return X_train, X_test, y_train, y_test


def get_cv():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def train_random_forest(X_train, y_train):
    print("\n--- Random Forest ---")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }
    rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
    grid = GridSearchCV(rf, param_grid, cv=get_cv(), scoring="f1", n_jobs=-1, verbose=1)
    t0 = time.time()
    grid.fit(X_train, y_train)
    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV F1:  {grid.best_score_:.4f}  ({time.time()-t0:.1f}s)")
    joblib.dump(grid.best_estimator_, f"{MODELS_DIR}/random_forest.pkl")
    return grid.best_estimator_


def train_svm(X_train, y_train):
    print("\n--- SVM (RBF kernel) ---")
    param_grid = {
        "C": [0.1, 1, 10],
        "gamma": ["scale", "auto"],
    }
    svm = SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)
    grid = GridSearchCV(svm, param_grid, cv=get_cv(), scoring="f1", n_jobs=-1, verbose=1)
    t0 = time.time()
    grid.fit(X_train, y_train)
    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV F1:  {grid.best_score_:.4f}  ({time.time()-t0:.1f}s)")
    joblib.dump(grid.best_estimator_, f"{MODELS_DIR}/svm.pkl")
    return grid.best_estimator_


def train_logistic_regression(X_train, y_train):
    print("\n--- Logistic Regression (baseline) ---")
    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["lbfgs", "liblinear"],
    }
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    grid = GridSearchCV(lr, param_grid, cv=get_cv(), scoring="f1", n_jobs=-1, verbose=1)
    t0 = time.time()
    grid.fit(X_train, y_train)
    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV F1:  {grid.best_score_:.4f}  ({time.time()-t0:.1f}s)")
    joblib.dump(grid.best_estimator_, f"{MODELS_DIR}/logistic_regression.pkl")
    return grid.best_estimator_


def run_training():
    print("Loading processed data...")
    X_train, X_test, y_train, y_test = load_processed_data()
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    models = {}
    models["Random Forest"] = train_random_forest(X_train, y_train)
    models["SVM"] = train_svm(X_train, y_train)
    models["Logistic Regression"] = train_logistic_regression(X_train, y_train)

    print("\nAll models trained and saved to models/")
    return models, X_test, y_test


if __name__ == "__main__":
    run_training()
