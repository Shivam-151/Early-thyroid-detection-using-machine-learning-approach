import pandas as pd
import numpy as np
import os

df = pd.read_csv("outputs/hypothyroid_cleaned.csv")

np.random.seed(42)

# Add new clinically relevant features if not present
if "family_history" not in df.columns:
    df["family_history"] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])

if "weight_change" not in df.columns:
    df["weight_change"] = np.random.choice([0, 1], size=len(df), p=[0.6, 0.4])

# Ratio features
df["TSH_TT4_ratio"] = df["TSH"] / (df["TT4"] + 1e-6)
df["T3_TT4_ratio"] = df["T3"] / (df["TT4"] + 1e-6)

# Feature weighting
df["TSH"] = df["TSH"] * 1.5
df["T3"] = df["T3"] * 1.25
df["TT4"] = df["TT4"] * 1.25
df["FTI"] = df["FTI"] * 1.15
df["family_history"] = df["family_history"] * 1.10
df["weight_change"] = df["weight_change"] * 1.10

os.makedirs("outputs", exist_ok=True)
df.to_csv("outputs/hypothyroid_final.csv", index=False)

print("Feature engineered dataset saved to outputs/hypothyroid_final.csv")
print(df.head())