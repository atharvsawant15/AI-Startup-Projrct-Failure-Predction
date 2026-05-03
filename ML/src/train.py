# Train logistic regression with scaling and custom threshold and return feature names for inference
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def train_model(data_path):
    df = pd.read_csv(data_path)
    df = pd.get_dummies(df, drop_first=True)

    print("data loaded:", df.shape)

    X = df.drop("failed", axis=1)
    y = df["failed"]

    # save feature names before scaling
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=2000)

    print("\ntraining model...")
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.4).astype(int)

    print("\n--- Evaluation ---")
    print("accuracy:", accuracy_score(y_test, y_pred))

    print("\nclassification report:")
    print(classification_report(y_test, y_pred))

    print("\nconfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, scaler, feature_names


if __name__ == "__main__":
    train_model("../data/processed/featured.csv")