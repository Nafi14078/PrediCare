import joblib
import pandas as pd
import numpy as np
import yaml
from typing import Dict, Any
from src.preprocessing.clean_data import preprocess_data


class InferencePipeline:
    def __init__(self, config_path='config.yaml'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.models = self._load_models()
        self.feature_engineering = None

    def _load_models(self) -> Dict[str, Any]:
        """Load all trained models"""
        models = {}
        for disease in ['heart_disease', 'diabetes', 'breast_cancer']:
            path = (self.config['models']['save_path'] +
                    self.config['models'][f'{disease}_model'])
            models[disease] = joblib.load(path)
        return models

    def preprocess_input(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data for prediction"""
        # Handle both file path and direct DataFrame input
        if isinstance(input_data, str):
            data = pd.read_csv(input_data)
        else:
            data = input_data.copy()

        # Basic cleaning
        data = data.dropna(axis=1, how='all')  # Drop completely empty columns

        # Feature engineering if available
        if self.feature_engineering:
            data = self.feature_engineering.transform(data)

        return data

    def predict_single_disease(self, model_name: str, input_data: pd.DataFrame) -> Dict:
        """Make prediction using a specific disease model"""
        model = self.models[model_name]

        # Preprocess data to match training format
        processed_data = self.preprocess_input(input_data)

        # Ensure columns match training data
        try:
            if hasattr(model, 'feature_names_in_'):
                processed_data = processed_data[model.feature_names_in_]

            proba = model.predict_proba(processed_data)
            prediction = model.predict(processed_data)

            # For binary classification, get probability of positive class
            confidence = proba[0][1] if len(proba[0]) > 1 else proba[0][0]

            return {
                'prediction': prediction[0],
                'confidence': float(confidence),
                'probabilities': proba.tolist()[0]
            }
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")

    def predict_all_diseases(self, input_data: pd.DataFrame) -> Dict:
        """Run prediction through all disease models"""
        results = {}

        for disease, model in self.models.items():
            try:
                results[disease] = self.predict_single_disease(disease, input_data)
            except Exception as e:
                results[disease] = {'error': str(e)}

        # Add combined interpretation
        if all('confidence' in res for res in results.values()):
            best_prediction = max(results.items(),
                                  key=lambda x: x[1]['confidence'])
            results['combined_prediction'] = {
                'most_likely': best_prediction[0],
                'confidence': best_prediction[1]['confidence']
            }

        return results

# Example usage:
# pipeline = InferencePipeline()
# result = pipeline.predict_all_diseases('path/to/new_data.csv')
# print(result)