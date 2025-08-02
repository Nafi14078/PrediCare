# src/deployment/predict_disease.py
import pandas as pd
import numpy as np
import joblib
import yaml
import os
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError


class DiseasePredictor:
    def __init__(self, config_path='config.yaml'):
        """Initialize predictor with models and feature requirements"""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.models = self._load_models()
        self.scalers = self._load_scalers()
        self.feature_requirements = {
            'heart_disease': [
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
                'ca', 'thal'
            ],
            'diabetes': [
                'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
            ],
            'breast_cancer': [f'feature_{i}' for i in range(1, 31)]  # 30 features
        }

    def _load_models(self):
        """Load all trained models from disk"""
        models = {}
        for disease in ['heart_disease', 'diabetes', 'breast_cancer']:
            path = os.path.join(
                self.config['models']['save_path'],
                self.config['models'][f'{disease}_model']
            )
            try:
                models[disease] = joblib.load(path)
                print(f"✅ Successfully loaded {disease} model")
            except Exception as e:
                print(f"❌ Error loading {disease} model: {str(e)}")
                raise
        return models

    def _load_scalers(self):
        """Load feature scalers used during training"""
        scalers = {}
        for disease in ['heart_disease', 'diabetes', 'breast_cancer']:
            path = os.path.join(
                self.config['models']['save_path'],
                f'{disease}_scaler.pkl'
            )
            try:
                scalers[disease] = joblib.load(path)
            except FileNotFoundError:
                raise ValueError(
                    f"Scaler not found for {disease}. "
                    "Please retrain models to generate scalers."
                )
        return scalers

    def _validate_features(self, df, disease_type):
        """Ensure input data has required features"""
        required = self.feature_requirements[disease_type]
        missing = [f for f in required if f not in df.columns]
        if missing:
            raise ValueError(
                f"Missing features for {disease_type}: {missing}\n"
                f"Required features: {required}"
            )
        return df[required]  # Return only required columns in correct order

    def _preprocess_data(self, df, disease_type):
        """Apply identical preprocessing as training"""
        try:
            # Select and validate features
            df = self._validate_features(df, disease_type)

            # Convert to numeric and handle missing values
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.fillna(df.mean())

            # Apply scaling
            X = self.scalers[disease_type].transform(df)
            return X

        except Exception as e:
            raise ValueError(f"Preprocessing failed: {str(e)}") from e

    def predict(self, data_path):
        """Make prediction on new data"""
        try:
            df = pd.read_csv(data_path)

            # Determine which disease type matches the input features
            matched_disease = None
            for disease in self.models.keys():
                if all(feat in df.columns for feat in self.feature_requirements[disease]):
                    matched_disease = disease
                    break

            if not matched_disease:
                available = list(df.columns)
                raise ValueError(
                    "Input data doesn't match any disease model requirements.\n"
                    "Available features: " + ", ".join(available) + "\n"
                                                                    "Required features for:\n"
                                                                    f"- Heart disease: {len(self.feature_requirements['heart_disease'])} features\n"
                                                                    f"- Diabetes: {len(self.feature_requirements['diabetes'])} features\n"
                                                                    f"- Breast cancer: {len(self.feature_requirements['breast_cancer'])} features"
                )

            # Preprocess and predict
            X = self._preprocess_data(df, matched_disease)
            model = self.models[matched_disease]
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0]

            # Format results
            result = {
                'disease': matched_disease.replace('_', ' '),
                'prediction': int(pred),
                'confidence': float(np.max(proba)),
                'probabilities': {
                    str(cls): float(prob) for cls, prob in enumerate(proba)
                }
            }

            # Special handling for heart disease (0-4 classes)
            if matched_disease == 'heart_disease':
                result['clinical_status'] = "Disease detected" if pred > 0 else "No disease"
                result['severity'] = int(pred)  # Original 0-4 prediction

            return result

        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}") from e


if __name__ == "__main__":
    # Test the predictor directly
    try:
        predictor = DiseasePredictor()
        test_files = {
            'heart': 'test_samples/heart_test.csv',
            'diabetes': 'test_samples/diabetes_test.csv',
            'cancer': 'test_samples/cancer_test.csv'
        }

        for name, path in test_files.items():
            print(f"\nTesting {name} prediction...")
            try:
                result = predictor.predict(path)
                print("Prediction Result:")
                for k, v in result.items():
                    if k == 'probabilities':
                        print(f"{k}:")
                        for cls, prob in v.items():
                            print(f"  Class {cls}: {prob:.2%}")
                    else:
                        print(f"{k}: {v}")
            except Exception as e:
                print(f"❌ {name} test failed: {str(e)}")

    except Exception as e:
        print(f"Fatal error: {str(e)}")