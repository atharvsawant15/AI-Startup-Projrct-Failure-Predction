# call training function and save model, scaler and feature names

import joblib
from train import train_model


def save_model():
    # train and get model + scaler + feature names
    model, scaler, feature_names = train_model("../data/processed/featured.csv")

    # save all
    joblib.dump(model, "../output/model.pkl")
    joblib.dump(scaler, "../output/scaler.pkl")
    joblib.dump(feature_names, "../output/features.pkl")

    print("model, scaler and features saved successfully")


if __name__ == "__main__":
    save_model()