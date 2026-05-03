# Load model, scaler, features and generate prediction with LLM explanation

import pandas as pd
import joblib
from src.llm_explainer import generate_explanation


# -----------------------------
# Load model artifacts
# -----------------------------
model = joblib.load("output/model.pkl")
scaler = joblib.load("output/scaler.pkl")
feature_names = joblib.load("output/features.pkl")


# -----------------------------
# Prediction function
# -----------------------------
def predict_startup(data_dict):
    df = pd.DataFrame([data_dict])
    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_names, fill_value=0)
    df_scaled = scaler.transform(df)

    prob_success = model.predict_proba(df_scaled)[0][1]
    prob_failure = 1 - prob_success

    pred = 1 if prob_failure > 0.4 else 0   # 1 = Failure, 0 = Success

    return pred, prob_failure


# -----------------------------
# Multiple test inputs
# -----------------------------
if __name__ == "__main__":

    test_data = [
        # Failure case
        {
            "Industry": 3,
            "Startup_Age": 4,
            "Funding_Amount": 14131778,
            "Number_of_Founders": 4,
            "Founder_Experience": 4,
            "Employees_Count": 341,
            "Revenue": 41922318.27,
            "Burn_Rate": 500000,
            "Market_Size": 2,
            "Business_Model": 1,
            "Product_Uniqueness_Score": 3,
            "Customer_Retention_Rate": 32.00,
            "Marketing_Expense": 45204378.16,
            "revenue_per_employee": 122579.878,
            "burn_to_revenue_ratio": 0.0119,
            "revenue_to_funding_ratio": 2.96,
            "marketing_efficiency": 0.92,
            "employees_per_founder": 68.2,
            "experience_per_founder": 0.8,
            "product_market_fit_score": 96.02,
            "profit_margin_proxy": 41422318.27,
            "funding_per_year": 2826355.6
        },
        # Success case 2
        {
            "Industry": 3,
            "Startup_Age": 8,
            "Funding_Amount": 18328419,
            "Number_of_Founders": 2,
            "Founder_Experience": 13,
            "Employees_Count": 581,
            "Revenue": 97866143,
            "Burn_Rate": 602731,
            "Market_Size": 1,
            "Business_Model": 0,
            "Product_Uniqueness_Score": 2,
            "Customer_Retention_Rate": 79.61,
            "Marketing_Expense": 987830,
            "revenue_per_employee": 168316.94,
            "burn_to_revenue_ratio": 0.00616,
            "revenue_to_funding_ratio": 5.34,
            "marketing_efficiency": 99.08,
            "employees_per_founder": 290.5,
            "experience_per_founder": 6.5,
            "product_market_fit_score": 159.22,
            "profit_margin_proxy": 97263312,
            "funding_per_year": 2291052.38
        }
    ]


    # -----------------------------
    # Run predictions + LLM analysis
    # -----------------------------
    for i, data in enumerate(test_data):
        pred, prob = predict_startup(data)

        print(f"\n==============================")
        print(f"Test Case {i+1}")
        print("==============================")

        print("Prediction (1 = Failure, 0 = Success):", pred)
        print("Failure Probability:", round(prob, 3))

        # 🔥 LLM explanation
        explanation = generate_explanation(data, pred, prob)

        print("\n--- AI Analysis ---")
        print(explanation)