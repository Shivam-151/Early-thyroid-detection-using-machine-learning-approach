import pandas as pd
import numpy as np

column_names = ["class", "t3_resin", "thyroxin", "triiodothyronine", "tsh", "max_tsh_diff"]

df = pd.read_csv("data/new-thyroid.data", header=None, names=column_names)

# Replace ? with NaN
df.replace("?", np.nan, inplace=True)

print("Missing values in each column:")
print(df.isnull().sum())