"""
train_model.py
---------------
TASK 1: Credit Scoring Model

Trains and compares three classification algorithms (Logistic Regression,
Decision Tree, Random Forest) to predict an individual's creditworthiness,
then evaluates them with Precision, Recall, F1-Score, and ROC-AUC.

Run:
    python generate_data.py     # only needed once, creates data/credit_data.csv
    python train_model.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    accuracy_score, confusion_matrix, classification_report, roc_curve
)
import joblib

RANDOM_STATE = 42
DATA_PATH = os.path.join("data", "credit_data.csv")
OUT_DIR = "outputs"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. Run `python generate_data.py` first, "
            f"or place your own dataset at this path with a 'creditworthy' target column."
        )
    return pd.read_csv(path)


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """A little extra feature engineering from the raw financial history,
    as required by the task ('Feature engineering from financial history')."""
    df = df.copy()
    df["income_to_loan_ratio"] = df["annual_income"] / (df["loan_amount_requested"] + 1)
    df["savings_to_income_ratio"] = df["savings_balance"] / (df["annual_income"] + 1)
    df["late_payments_per_credit_year"] = df["num_late_payments"] / (df["credit_history_length_years"] + 1)
    return df


def evaluate_model(name, model, X_test, y_test, results, out_dir=OUT_DIR):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    results[name] = metrics

    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        print(f"{k:>10}: {v:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Not Creditworthy", "Creditworthy"]))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Creditworthy", "Creditworthy"],
                yticklabels=["Not Creditworthy", "Creditworthy"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"confusion_matrix_{name.replace(' ', '_')}.png"), dpi=150)
    plt.close()

    return y_proba


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_data()
    df = feature_engineer(df)

    X = df.drop(columns=["creditworthy"])
    y = df["creditworthy"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10, random_state=RANDOM_STATE),
    }

    results = {}
    plt.figure(figsize=(6, 5))

    for name, model in models.items():
        # Logistic Regression benefits from scaled features; tree-based
        # models are scale-invariant but scaling doesn't hurt them.
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            y_proba = evaluate_model(name, model, X_test_scaled, y_test, results)
        else:
            model.fit(X_train, y_train)
            y_proba = evaluate_model(name, model, X_test, y_test, results)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.3f})")

        joblib.dump(model, os.path.join(OUT_DIR, f"model_{name.replace(' ', '_')}.pkl"))

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "roc_curves.png"), dpi=150)
    plt.close()

    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))

    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    best_model = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"\nBest model by ROC-AUC: {best_model} ({results[best_model]['roc_auc']:.4f})")
    print(f"\nAll metrics, plots and trained models saved to '{OUT_DIR}/'")

    # Feature importance for Random Forest (helps explain the model)
    rf = models["Random Forest"]
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(7, 5))
    sns.barplot(x=importances.values[:10], y=importances.index[:10])
    plt.title("Top 10 Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "feature_importance.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
