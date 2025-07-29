import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.preprocessing.clean_data import clean_all_data

# Load and clean datasets
heart_df, diabetes_df, breast_df = clean_all_data()

# Select which dataset to evaluate
df = heart_df  # You can change to diabetes_df or breast_df

X = df.drop("target", axis=1)
y = df["target"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the trained model
model_path = os.path.join("saved_models", "disease_classifier.pkl")
model = joblib.load(model_path)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluation
print("\n[INFO] Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n[INFO] Classification Report:")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"\n[INFO] Accuracy Score: {accuracy:.4f}")
