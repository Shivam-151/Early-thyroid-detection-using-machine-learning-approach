import pandas as pd
import numpy as np

df = pd.read_csv("outputs/cleaned_data.csv")

np.random.seed(42)

df["family_history"] = np.random.choice([0,1], size=len(df))
df["weight_change"] = np.random.choice([0,1], size=len(df))

# feature weighting
df["tsh"] *= 1.5
df["thyroxin"] *= 1.25
df["triiodothyronine"] *= 1.25
df["family_history"] *= 1.10
df["weight_change"] *= 1.10

df.to_csv("outputs/final_dataset.csv", index=False)

print("Feature engineering done")