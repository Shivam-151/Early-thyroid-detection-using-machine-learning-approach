import pandas as pd
import numpy as np
import os

columns = [
    "class",
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
    "TBG"
]

df = pd.read_csv(
    "data/hypothyroid.data",
    header=None,
    names=columns,
    na_values=["?"]
)

print("Original shape:", df.shape)

# remove duplicates
df.drop_duplicates(inplace=True)

# clean class text
df["class"] = df["class"].astype(str).str.strip().str.lower()

# keep binary classes only
df = df[df["class"].isin(["hypothyroid", "negative"])].copy()

# numeric columns
numeric_cols = ["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# fill numeric missing values with median
for col in numeric_cols:
    if df[col].isnull().all():
        df[col] = df[col].fillna(0)
    else:
        df[col] = df[col].fillna(df[col].median())

# clean sex
df["sex"] = df["sex"].astype(str).str.strip().str.upper()
df["sex"] = df["sex"].replace("NAN", np.nan)
if df["sex"].mode().empty:
    df["sex"] = df["sex"].fillna("F")
else:
    df["sex"] = df["sex"].fillna(df["sex"].mode().iloc[0])

# map sex
df["sex"] = df["sex"].map({"F": 0, "M": 1})
df["sex"] = df["sex"].fillna(0)

# these columns are f/t
ft_cols = [
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
    "goitre"
]

for col in ft_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()
    df[col] = df[col].map({"f": 0, "t": 1})
    df[col] = df[col].fillna(0)

# these columns are y/n
yn_cols = [
    "TSH_measured",
    "T3_measured",
    "TT4_measured",
    "T4U_measured",
    "FTI_measured",
    "TBG_measured"
]

for col in yn_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()
    df[col] = df[col].map({"n": 0, "y": 1})
    df[col] = df[col].fillna(0)

os.makedirs("outputs", exist_ok=True)
df.to_csv("outputs/hypothyroid_cleaned_correct.csv", index=False)

print("Cleaned shape:", df.shape)
print("\nClass counts:")
print(df["class"].value_counts())
print("\nMissing values after cleaning:")
print(df.isnull().sum())
print("\nSaved: outputs/hypothyroid_cleaned_correct.csv")