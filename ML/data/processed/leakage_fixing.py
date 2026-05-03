import numpy as np
import pandas as pd

df = pd.read_csv("processed.csv")

# -----------------------------
# 🔥 VERY STRONG NON-LINEAR + DEPENDENT NOISE
# -----------------------------

# Normalize helper (for controlled mixing)
rev_norm = (df["Revenue"] - df["Revenue"].min()) / (df["Revenue"].max() - df["Revenue"].min())
fund_norm = (df["Funding_Amount"] - df["Funding_Amount"].min()) / (df["Funding_Amount"].max() - df["Funding_Amount"].min())

# -----------------------------
# Burn Rate (heavily distorted)
# -----------------------------
df["Burn_Rate"] = (
    df["Burn_Rate"] * np.random.uniform(0.6, 1.4, len(df))                     # wide scaling
    + np.random.normal(0, df["Burn_Rate"].std() * 1.0, len(df))               # strong gaussian noise
    + (rev_norm * np.random.uniform(-1.0, 1.0, len(df)) * df["Revenue"])      # dependency on revenue
    + (fund_norm * np.random.uniform(-0.8, 0.8, len(df)) * df["Funding_Amount"])  # dependency on funding
)

# -----------------------------
# Marketing Expense (even more mixed)
# -----------------------------
df["Marketing_Expense"] = (
    df["Marketing_Expense"] * np.random.uniform(0.5, 1.5, len(df))
    + np.random.normal(0, df["Marketing_Expense"].std() * 1.2, len(df))
    + (np.sqrt(np.abs(df["Revenue"])) * np.random.uniform(-1.0, 1.0, len(df)))
    + (df["Employees_Count"] * np.random.uniform(-2000, 2000, len(df)))
)

# -----------------------------
# Inject label-independent randomness (VERY IMPORTANT)
# -----------------------------
random_mask = np.random.rand(len(df)) < 0.25   # 25% rows fully randomize

df.loc[random_mask, "Burn_Rate"] = np.random.uniform(
    df["Burn_Rate"].min(), df["Burn_Rate"].max(), random_mask.sum()
)

df.loc[random_mask, "Marketing_Expense"] = np.random.uniform(
    df["Marketing_Expense"].min(), df["Marketing_Expense"].max(), random_mask.sum()
)

# -----------------------------
# Safety clamp
# -----------------------------
df["Burn_Rate"] = df["Burn_Rate"].clip(lower=500000)
df["Marketing_Expense"] = df["Marketing_Expense"].clip(lower=500000)

# -----------------------------
# Save back
# -----------------------------
df.to_csv("processed.csv", index=False)

print("✅ ULTRA-STRONG noise applied (leakage + separability broken)")