import pandas as pd
import os


def create_features(input_path, output_path):
    # read the already cleaned dataset
    df = pd.read_csv(input_path)
    print("loaded data:", df.shape)

    # ---------- basic efficiency metrics ----------

    # how productive each employee is
    df["revenue_per_employee"] = df["Revenue"] / (df["Employees_Count"] + 1)

    # how much you're burning compared to what you earn
    df["burn_to_revenue_ratio"] = df["Burn_Rate"] / (df["Revenue"] + 1)

    # returns generated from total funding
    df["revenue_to_funding_ratio"] = df["Revenue"] / (df["Funding_Amount"] + 1)

    # are we overspending on marketing or actually getting returns?
    df["marketing_efficiency"] = df["Revenue"] / (df["Marketing_Expense"] + 1)

    # ---------- team related signals ----------

    # too many employees with few founders? or balanced team?
    df["employees_per_founder"] = df["Employees_Count"] / (df["Number_of_Founders"] + 1)

    # how experienced the founders are overall
    df["experience_per_founder"] = df["Founder_Experience"] / (df["Number_of_Founders"] + 1)

    # ---------- product & user strength ----------

    # combining uniqueness + retention → strong indicator of product-market fit
    df["product_market_fit_score"] = (
        df["Product_Uniqueness_Score"] * df["Customer_Retention_Rate"]
    )

    # ---------- financial health ----------

    # simple profit-like signal (not exact profit but gives idea)
    df["profit_margin_proxy"] = df["Revenue"] - df["Burn_Rate"]

    # how much funding is used over time (growth pace)
    df["funding_per_year"] = df["Funding_Amount"] / (df["Startup_Age"] + 1)

    # ---------- save file ----------

    # create folder if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # write updated dataset
    df.to_csv(output_path, index=False)

    print("features added + saved at:", output_path)

    return df


if __name__ == "__main__":
    input_file = "../data/processed/processed.csv"
    output_file = "../data/processed/featured.csv"

    df = create_features(input_file, output_file)

    # quick check
    print(df.head())