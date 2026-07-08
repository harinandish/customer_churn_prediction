"""
train.py
Full ML pipeline for Customer Churn Prediction:
- Loads dataset
- Preprocesses demographic, usage, and engagement features
- Compares multiple models (Logistic Regression, Random Forest, Gradient Boosting)
- Uses cross-validation for model selection
- Evaluates the best model with ROC/AUC and classification metrics
- Saves the final trained pipeline
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

DATA_PATH = "data/customer_churn.csv"
MODEL_PATH = "models/churn_model.pkl"
OUTPUT_DIR = "output"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df.drop(columns=["customer_id"])
    return df


def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = load_data()

    X = df.drop(columns=["churn"])
    y = df["churn"]

    categorical_features = ["gender", "region", "subscription_type"]
    numeric_features = [c for c in X.columns if c not in categorical_features]

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_model_name = None
    best_score = -np.inf
    results = {}

    print("=== Cross-Validation Results (ROC-AUC) ===")
    for name, model in candidates.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        mean_score = scores.mean()
        results[name] = mean_score
        print(f"{name}: {mean_score:.4f} (+/- {scores.std():.4f})")

        if mean_score > best_score:
            best_score = mean_score
            best_model_name = name

    print(f"\nBest model from cross-validation: {best_model_name} (AUC={best_score:.4f})")

    # Refit best model on full training set
    final_pipe = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", candidates[best_model_name])]
    )
    final_pipe.fit(X_train, y_train)

    # Evaluate on held-out test set
    y_pred = final_pipe.predict(X_test)
    y_proba = final_pipe.predict_proba(X_test)[:, 1]

    test_auc = roc_auc_score(y_test, y_proba)
    print(f"\nTest set ROC-AUC: {test_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Retained", "Churned"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Customer Churn Prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()

    # Save ROC curve plot
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {test_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Customer Churn Prediction")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"))
    plt.close()

    # Save the trained pipeline
    joblib.dump(final_pipe, MODEL_PATH)
    print(f"\nSaved best model ('{best_model_name}') to {MODEL_PATH}")
    print(f"Saved evaluation plots to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
