import pandas as pd
import numpy as np
import os

columns = [
    "age",
    "sex",
    "on_thyroxine",
    "query_on_thyroxine",
    "on_antithyroid_medication",
    "thyroid_surgery",
    "query_hypothyroid",
    "query_hyperthyroid",
    "pregnant",
    "sick",
    "tumor",
    "lithium",
    "goitre",
    "TSH_measured",
    "TSH",
    "T3_measured",
    "T3",
    "TT4_measured",
    "TT4",
    "T4U_measured",
    "T4U",
    "FTI_measured",
    "FTI",
    "TBG_measured",
    "TBG",
    "class"
]

# Load dataset
df = pd.read_csv(
    "data/hypothyroid.data",
    header=None,
    names=columns,
    na_values=["?"]
)

print("Original shape:", df.shape)
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

# Clean class column
df["class"] = df["class"].astype(str).str.strip().str.lower().str.replace(".", "", regex=False)

# Convert to binary classes
df["class"] = df["class"].apply(
    lambda x: "hypothyroid" if "hypothyroid" in x else "negative"
)

# Numeric columns
numeric_cols = ["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

# Clean sex column
df["sex"] = df["sex"].astype(str).str.strip().str.upper()
df["sex"] = df["sex"].replace("NAN", np.nan)

if df["sex"].mode().empty:
    df["sex"] = df["sex"].fillna("F")
else:
    df["sex"] = df["sex"].fillna(df["sex"].mode().iloc[0])

# Binary columns
binary_cols = [
    "on_thyroxine",
    "query_on_thyroxine",
    "on_antithyroid_medication",
    "thyroid_surgery",
    "query_hypothyroid",
    "query_hyperthyroid",
    "pregnant",
    "sick",
    "tumor",
    "lithium",
    "goitre",
    "TSH_measured",
    "T3_measured",
    "TT4_measured",
    "T4U_measured",
    "FTI_measured",
    "TBG_measured"
]

for col in binary_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()
    df[col] = df[col].map({"f": 0, "t": 1})

# Encode sex
df["sex"] = df["sex"].map({"F": 0, "M": 1})

if df["sex"].mode().empty:
    df["sex"] = df["sex"].fillna(0)
else:
    df["sex"] = df["sex"].fillna(df["sex"].mode().iloc[0])

# Save cleaned dataset
os.makedirs("outputs", exist_ok=True)
df.to_csv("outputs/hypothyroid_cleaned.csv", index=False)

print("\nClass distribution after binary conversion:")
print(df["class"].value_counts())

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nCleaned dataset saved to outputs/hypothyroid_cleaned.csv")
print("Cleaned shape:", df.shape)