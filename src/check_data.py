import pandas as pd

columns = [
"class",
"t3_resin",
"thyroxin",
"triiodothyronine",
"tsh",
"max_tsh_diff"
]

df = pd.read_csv("data/new-thyroid.data", header=None, names=columns)


print("Shape:", df.shape)
print(df.head())