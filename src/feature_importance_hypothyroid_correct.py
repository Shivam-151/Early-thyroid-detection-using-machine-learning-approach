import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("outputs/hypothyroid_final_correct.csv")

X = df.drop("class", axis=1)
y = df["class"]

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_scaled, y_train)

importance = rf.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 8))
plt.barh(feature_names, importance)
plt.title("Feature Importance - Hypothyroid")
plt.tight_layout()
plt.savefig("outputs/hypothyroid_feature_importance_correct.png")
plt.show()

print("Saved: outputs/hypothyroid_feature_importance_correct.png")