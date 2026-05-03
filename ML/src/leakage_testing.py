# 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def leakage_detection_pipeline(file_path, target_col="failed"):
    df = pd.read_csv(file_path)
    print("\n📊 Dataset shape:", df.shape)

    # -----------------------------
    # 1. Correlation check
    # -----------------------------
    print("\n🔍 Correlation with target:\n")
    corr = df.corr(numeric_only=True)[target_col].sort_values(ascending=False)
    print(corr)

    print("\n🚨 High correlation features (>0.8):")
    high_corr = corr[abs(corr) > 0.8].drop(target_col, errors="ignore")
    print(high_corr)

    # -----------------------------
    # 2. Duplicate rows check
    # -----------------------------
    dup_count = df.duplicated().sum()
    print("\n🔍 Duplicate rows:", dup_count)
    if dup_count > 0:
        print("🚨 WARNING: Duplicates may inflate accuracy!")

    # -----------------------------
    # 3. Single feature leakage test
    # -----------------------------
    print("\n🔍 Single Feature Leakage Test:\n")

    y = df[target_col]

    for col in df.columns:
        if col == target_col:
            continue

        X = df[[col]]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))

        if acc > 0.90:
            print(f"🚨 LEAKAGE SUSPECT: {col} -> Accuracy: {acc:.4f}")
        else:
            print(f"OK: {col} -> Accuracy: {acc:.4f}")

    # -----------------------------
    # 4. Random label sanity check
    # -----------------------------
    print("\n🎲 Random Label Test:")

    X = df.drop(columns=[target_col])
    y_random = np.random.permutation(df[target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_random, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    print("Random label accuracy:", acc)

    if acc > 0.6:
        print("🚨 WARNING: Model still finds patterns → possible leakage or synthetic bias")
    else:
        print("✅ OK: No strong leakage detected")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    leakage_detection_pipeline("../data/processed/featured.csv")