import pandas as pd

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

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())
print("\nClass counts:")
print(df["class"].value_counts())