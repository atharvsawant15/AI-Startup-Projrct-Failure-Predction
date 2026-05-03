import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder


def preprocess_data(input_path, output_path):
    # load the csv file
    df = pd.read_csv(input_path)
    print("data loaded:", df.shape)

    # drop startup name since it's just an identifier (not useful for ML)
    if "Startup_Name" in df.columns:
        df.drop("Startup_Name", axis=1, inplace=True)

    # separate numeric and categorical columns
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    # fill missing numeric values with median (safer than mean for skewed data)
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # fill missing categorical values with most frequent value
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # convert target column to numeric if it's in text format
    if df["Startup_Status"].dtype == "object":
        df["Startup_Status"] = df["Startup_Status"].map({
            "Failed": 1,
            "Success": 0
        })

    # rename target column for simplicity
    df.rename(columns={"Startup_Status": "failed"}, inplace=True)

    # encode categorical features (industry, market size, business model etc.)
    encoders = {}
    cat_cols = df.select_dtypes(include=["object"]).columns

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # make sure output folder exists before saving
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # save cleaned dataset
    df.to_csv(output_path, index=False)
    print("processed data saved at:", output_path)

    return df, encoders


if __name__ == "__main__":
    input_file = "../data/raw/final_balanced_dataset.csv"
    output_file = "../data/processed/processed.csv"

    df, encoders = preprocess_data(input_file, output_file)

    # quick preview to verify everything looks fine
    print(df.head())

    raw_df = pd.read_csv(input_file)
    print(raw_df["Startup_Status"].value_counts())