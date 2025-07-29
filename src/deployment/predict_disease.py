# src/deployment/predict_disease.py

import joblib
import numpy as np
import pandas as pd

def load_model(model_path='saved_models/disease_classifier.pkl'):
    model = joblib.load(model_path)
    return model

def predict(model, input_data):
    # input_data should be a Pandas DataFrame with correct features
    preds = model.predict(input_data)
    return preds

if __name__ == "__main__":
    print("Load your trained model and pass patient data to predict.")
