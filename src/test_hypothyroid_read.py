import pandas as pd

df = pd.read_csv("data/hypothyroid.data", header=None, sep=",", engine="python")

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nNon-null count per column:")
print(df.notnull().sum())

print("\nLast columns preview:")
print(df.iloc[:, -5:].head())