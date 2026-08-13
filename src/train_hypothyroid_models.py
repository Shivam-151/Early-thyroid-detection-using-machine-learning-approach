import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load final dataset
df = pd.read_csv("outputs/hypothyroid_final.csv")

# Force class cleanup again for safety
df["class"] = df["class"].astype(str).str.strip().str.lower()
df["class"] = df["class"].apply(lambda x: "hypothyroid" if "hypothyroid" in x else "negative")

print("Training file loaded successfully")
print("Shape:", df.shape)
print("\nClass counts after forced conversion:")
print(df["class"].value_counts())

# Separate features and target
X = df.drop("class", axis=1)
y = df["class"]

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
rf = RandomForestClassifier(n_estimators=200, random_state=42)
svm = SVC(probability=True, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
ann = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42)

# Train
rf.fit(X_train_scaled, y_train)
svm.fit(X_train_scaled, y_train)
knn.fit(X_train_scaled, y_train)
ann.fit(X_train_scaled, y_train)

# Ensemble
ensemble = VotingClassifier(
    estimators=[
        ("rf", rf),
        ("svm", svm),
        ("ann", ann)
    ],
    voting="soft"
)

ensemble.fit(X_train_scaled, y_train)

# Evaluate
models = {
    "Random Forest": rf,
    "SVM": svm,
    "KNN": knn,
    "ANN": ann,
    "Ensemble": ensemble
}

os.makedirs("outputs", exist_ok=True)
with open("outputs/hypothyroid_results.txt", "w", encoding="utf-8") as f:
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)

        print(f"\n{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

        f.write(f"\n{name} Accuracy: {acc:.4f}\n")
        f.write(classification_report(y_test, y_pred))
        f.write("\n" + "=" * 60 + "\n")

# Save model and scaler
os.makedirs("models", exist_ok=True)
joblib.dump(ensemble, "models/hypothyroid_best_model.pkl")
joblib.dump(scaler, "models/hypothyroid_scaler.pkl")

print("\nModel saved to models/hypothyroid_best_model.pkl")
print("Scaler saved to models/hypothyroid_scaler.pkl")
print("Results saved to outputs/hypothyroid_results.txt")