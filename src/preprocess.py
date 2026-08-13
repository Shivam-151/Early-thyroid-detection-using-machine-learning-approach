import pandas as pd
import numpy as np

columns = [
"class",
"t3_resin",
"thyroxin",
"triiodothyronine",
"tsh",
"max_tsh_diff"
]

df = pd.read_csv("data/new-thyroid.data", header=None, names=columns)

# replace missing values
df.replace("?", np.nan, inplace=True)

# convert numeric columns
for col in df.columns:
    if col != "class":
        df[col] = pd.to_numeric(df[col], errors="coerce")

# fill missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# remove duplicates
df.drop_duplicates(inplace=True)

df.to_csv("outputs/cleaned_data.csv", index=False)

print("Cleaning complete")