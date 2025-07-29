# src/pipeline/training_pipeline.py

import os
import joblib
import pandas as pd
from src.preprocessing.clean_data import clean_all_data
from src.models.model_selector import train_and_select_model

# Load and clean datasets
heart_df, diabetes_df, breast_df = clean_all_data()

# You can choose which dataset to train on for now
X = heart_df.drop("target", axis=1)
y = heart_df["target"]

# Train and select best model
best_model, model_name = train_and_select_model(X, y)

# Save the model
model_path = os.path.join("saved_models", "disease_classifier.pkl")
joblib.dump(best_model, model_path)

print(f"[INFO] Best model ({model_name}) saved to: {model_path}")
