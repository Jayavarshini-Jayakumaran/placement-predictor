"""
train_model.py
---------------
Trains the Campus Placement Eligibility model end-to-end:

1. Data preprocessing  (missing value / duplicate checks, scaling)
2. Feature engineering (experience_score, cgpa_band, interaction term)
3. Train/test split
4. Model training       (RandomForestClassifier)
5. Evaluation           (accuracy, precision, recall, F1,
                          confusion matrix, ROC-AUC, classification report)
6. Artifacts saved:
       - model.pkl      trained model
       - scaler.pkl     fitted StandardScaler

Run with:
    python train_model.py
"""

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from features import add_engineered_features, FEATURE_COLUMNS


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        print(f"Removing {n_dupes} duplicate row(s)...")
        df = df.drop_duplicates().reset_index(drop=True)

    n_missing = df.isnull().sum().sum()
    if n_missing > 0:
        print(f"Found {n_missing} missing value(s). Filling with column medians...")
        df = df.fillna(df.median(numeric_only=True))
    else:
        print("No missing values found.")

    return df


def main():
    df = pd.read_csv("data.csv")
    print(f"Loaded data.csv  —  {df.shape[0]} rows, {df.shape[1]} columns")

    df = preprocess(df)
    df = add_engineered_features(df)

    X = df[FEATURE_COLUMNS]
    y = df["eligible"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]} samples  |  Test: {X_test.shape[0]} samples")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train_scaled, y_train)

    y_pred  = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"F1-score : {f1_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.3f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("Saved model.pkl and scaler.pkl  —  Model trained!")


if __name__ == "__main__":
    main()
