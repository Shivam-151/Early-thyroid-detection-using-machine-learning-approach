import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score

df = pd.read_csv("outputs/final_dataset.csv")

X = df.drop("class", axis=1)
y = df["class"]

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
X,y,test_size=0.2,random_state=42,stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf = RandomForestClassifier()
svm = SVC(probability=True)
knn = KNeighborsClassifier()
ann = MLPClassifier(max_iter=500)

rf.fit(X_train,y_train)
svm.fit(X_train,y_train)
knn.fit(X_train,y_train)
ann.fit(X_train,y_train)

print("RF:",accuracy_score(y_test,rf.predict(X_test)))
print("SVM:",accuracy_score(y_test,svm.predict(X_test)))
print("KNN:",accuracy_score(y_test,knn.predict(X_test)))
print("ANN:",accuracy_score(y_test,ann.predict(X_test)))

from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
estimators=[
("rf",rf),
("svm",svm),
("ann",ann)
],
voting="soft"
)

ensemble.fit(X_train,y_train)

print("Ensemble accuracy:",
accuracy_score(y_test,ensemble.predict(X_test)))

import joblib
import os

os.makedirs("models", exist_ok=True)

joblib.dump(ensemble, "models/best_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model and scaler saved successfully")