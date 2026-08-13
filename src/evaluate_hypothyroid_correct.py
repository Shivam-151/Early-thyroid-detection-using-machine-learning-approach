import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

scaler = joblib.load("models/hypothyroid_scaler_correct.pkl")
model = joblib.load("models/hypothyroid_best_model_correct.pkl")

X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Hypothyroid Ensemble")
plt.savefig("outputs/hypothyroid_confusion_matrix_correct.png")
plt.show()

print("Saved: outputs/hypothyroid_confusion_matrix_correct.png")