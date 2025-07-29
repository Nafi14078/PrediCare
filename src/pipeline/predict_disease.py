import joblib
import pandas as pd

# Load the saved model
model = joblib.load("saved_models/disease_classifier.pkl")

# Define a sample new patient input (must match feature columns of heart dataset)
new_patient = pd.DataFrame([{
    'age': 60,
    'sex': 1,
    'cp': 2,
    'trestbps': 140,
    'chol': 289,
    'fbs': 0,
    'restecg': 1,
    'thalach': 172,
    'exang': 0,
    'oldpeak': 0.0,
    'slope': 2,
    'ca': 0,
    'thal': 2
}])

# Predict
prediction = model.predict(new_patient)[0]
print(f"[INFO] Prediction for Heart Disease: {'Yes' if prediction == 1 else 'No'}")
