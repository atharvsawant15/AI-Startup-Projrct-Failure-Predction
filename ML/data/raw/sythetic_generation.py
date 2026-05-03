import pandas as pd
import numpy as np

np.random.seed(42)

n_fail = 4500

# ----------------------------
# FAILED STARTUP DATA
# ----------------------------
fail_df = pd.DataFrame({
    "Startup_Name": [f"Startup_F_{i}" for i in range(n_fail)],
    "Industry": np.random.choice(["Tech","Health","Finance","Logistics"], n_fail),
    "Startup_Age": np.random.randint(1, 10, n_fail),
    "Funding_Amount": np.random.randint(1_000_000, 40_000_000, n_fail),
    "Number_of_Founders": np.random.randint(1, 5, n_fail),
    "Founder_Experience": np.random.randint(1, 12, n_fail),
    "Employees_Count": np.random.randint(10, 800, n_fail),
    "Revenue": np.random.randint(2_000_000, 80_000_000, n_fail),
    "Burn_Rate": np.random.randint(2_000_000, 50_000_000, n_fail),
    "Market_Size": np.random.choice(["Small","Medium","Large"], n_fail),
    "Business_Model": np.random.choice(["B2B","B2C"], n_fail),
    "Product_Uniqueness_Score": np.random.randint(1, 8, n_fail),
    "Customer_Retention_Rate": np.random.uniform(30, 85, n_fail),
    "Marketing_Expense": np.random.randint(2_000_000, 30_000_000, n_fail),
    "Startup_Status": 0
})

# ----------------------------
# SAFE REALISTIC NOISE (FAILED DATA)
# ----------------------------

# Revenue noise (already OK)
fail_df["Revenue"] = fail_df["Revenue"] * np.random.uniform(0.7, 1.3, n_fail)

# 🔥 FIXED: Burn Rate (add gaussian noise instead of scaling only)
fail_df["Burn_Rate"] = fail_df["Burn_Rate"] + np.random.normal(0, 2_000_000, n_fail)

# 🔥 FIXED: Marketing Expense (IMPORTANT leakage fix)
fail_df["Marketing_Expense"] = fail_df["Marketing_Expense"] + np.random.normal(0, 1_500_000, n_fail)

# Customer retention noise
fail_df["Customer_Retention_Rate"] += np.random.uniform(-10, 10, n_fail)
fail_df["Customer_Retention_Rate"] = fail_df["Customer_Retention_Rate"].clip(20, 95)

# safety: no negative values
fail_df["Burn_Rate"] = fail_df["Burn_Rate"].clip(lower=500_000)
fail_df["Marketing_Expense"] = fail_df["Marketing_Expense"].clip(lower=500_000)

fail_df.to_csv("failed_startups_4500.csv", index=False)
print("failed dataset saved")

# ----------------------------
# SUCCESS DATA
# ----------------------------
df_success = pd.read_csv("success_startups_5000.csv")

df_success["Revenue"] = df_success["Revenue"] * np.random.uniform(0.8, 1.2, len(df_success))
df_success["Burn_Rate"] = df_success["Burn_Rate"] * np.random.uniform(0.7, 1.3, len(df_success))
df_success["Customer_Retention_Rate"] += np.random.uniform(-5, 5, len(df_success))
df_success["Customer_Retention_Rate"] = df_success["Customer_Retention_Rate"].clip(30, 100)

# 🔥 ALSO FIX SUCCESS SIDE (VERY IMPORTANT)
df_success["Marketing_Expense"] = df_success["Marketing_Expense"] + np.random.normal(0, 1_000_000, len(df_success))
df_success["Marketing_Expense"] = df_success["Marketing_Expense"].clip(lower=500_000)

df_success["Burn_Rate"] = df_success["Burn_Rate"] + np.random.normal(0, 1_500_000, len(df_success))
df_success["Burn_Rate"] = df_success["Burn_Rate"].clip(lower=500_000)

# ----------------------------
# COMBINE DATASETS
# ----------------------------
df_final = pd.concat([df_success, fail_df], ignore_index=True)

# shuffle
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

df_final.to_csv("final_balanced_dataset.csv", index=False)

print("final dataset saved")
print(df_final["Startup_Status"].value_counts())