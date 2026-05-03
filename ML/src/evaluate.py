# Evaluate trained model using graphs like confusion matrix, ROC curve and precision-recall curve

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve
)


def evaluate_model(data_path):
    # load dataset
    df = pd.read_csv(data_path)
    df = pd.get_dummies(df, drop_first=True)

    # split features and target
    X = df.drop("failed", axis=1)
    y = df["failed"]

    # load trained model and scaler
    model = joblib.load("../output/model.pkl")
    scaler = joblib.load("../output/scaler.pkl")

    # scale features
    X_scaled = scaler.transform(X)

    # get predictions
    y_prob = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_prob > 0.4).astype(int)

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

    # -----------------------------
    # ROC Curve
    # -----------------------------
    fpr, tpr, _ = roc_curve(y, y_prob)
    auc = roc_auc_score(y, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1])  # baseline
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    # -----------------------------
    # Precision-Recall Curve
    # -----------------------------
    precision, recall, _ = precision_recall_curve(y, y_prob)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.show()


if __name__ == "__main__":
    evaluate_model("../data/processed/processed.csv")