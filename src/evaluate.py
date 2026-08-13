import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load dataset again
df = pd.read_csv("outputs/final_dataset.csv")

# Separate features and target
X = df.drop("class", axis=1)
y = df["class"]

# Encode target column
le = LabelEncoder()
y = le.fit_transform(y)

# Train-test split again (same parameters as before)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load scaler and model
scaler = joblib.load("models/scaler.pkl")
model = joblib.load("models/best_model.pkl")

# Scale test data
X_test_scaled = scaler.transform(X_test)

# Predict
y_pred = model.predict(X_test_scaled)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix - Ensemble Model")
plt.show()