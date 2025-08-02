# src/pipeline/training_pipeline.py
import yaml
from sklearn.model_selection import train_test_split
from src.models.model_selector import ModelSelector
from src.preprocessing.clean_data import load_data, preprocess_data
import joblib
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np
from imblearn.over_sampling import SMOTE
import datetime  # For versioning


class TrainingPipeline:
    def __init__(self, config_path='config.yaml'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.selector = ModelSelector(config_path)

    def _balance_classes(self, X, y):
        """Handle class imbalance using SMOTE"""
        print(f"Class distribution before balancing: {np.bincount(y)}")
        if len(np.unique(y)) > 1:  # Only balance if multiple classes exist
            X, y = SMOTE().fit_resample(X, y)
            print(f"Class distribution after balancing: {np.bincount(y)}")
        return X, y

    def _evaluate_model(self, model, X_test, y_test):
        """Generate comprehensive evaluation report"""
        y_pred = model.predict(X_test)
        print("\nModel Evaluation Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']))

    def _cleanup_old_models(self, disease_type):
        """Remove old versions of models"""
        import glob
        pattern = os.path.join(
            self.config['models']['save_path'],
            f'{disease_type}_model_*.pkl'
        )
        for old_model in glob.glob(pattern):
            try:
                os.remove(old_model)
                print(f"Removed old model: {old_model}")
            except Exception as e:
                print(f"Error removing {old_model}: {str(e)}")

    def train_for_disease(self, disease_type):
        # Add this at the start:
        self._cleanup_old_models(disease_type)
        """Complete training pipeline for a single disease"""
        filepath = self.config['data'][disease_type]

        try:
            # 1. Load and preprocess data
            print(f"\nLoading {disease_type} data...")
            df = load_data(filepath)
            target_col = df.attrs.get('target_column', df.columns[-1])
            X, y = preprocess_data(df, target_col)

            # 2. Handle class imbalance
            X, y = self._balance_classes(X, y)

            # 3. Train-test split with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )

            # 4. Save feature scaler
            scaler = StandardScaler().fit(X_train)
            scaler_path = os.path.join(
                self.config['models']['save_path'],
                f'{disease_type}_scaler.pkl'
            )
            joblib.dump(scaler, scaler_path)
            print(f"Scaler saved to {scaler_path}")

            # 5. Train model
            print("Training model...")
            model = self.selector.select_best_model(X_train, y_train)

            # 6. Evaluate model
            self._evaluate_model(model, X_test, y_test)

            # 7. Save model with versioning
            version = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            # Change the saving code to:
            model_path = os.path.join(
                self.config['models']['save_path'],
                self.config['models'][f'{disease_type}_model']  # Use config name
            )
            joblib.dump(model, model_path)
            print(f"Model saved to {model_path}")

            return model

        except Exception as e:
            print(f"\nError in {disease_type} training pipeline: {str(e)}")
            raise

    def train_all(self):
        """Orchestrate training for all diseases"""
        print("Starting training pipeline...")
        diseases = ['heart_disease', 'diabetes', 'breast_cancer']
        models = {}

        for disease in diseases:
            print(f"\n{'=' * 40}")
            print(f"Training {disease.replace('_', ' ')} model")
            print(f"{'=' * 40}")
            try:
                models[disease] = self.train_for_disease(disease)
            except Exception as e:
                print(f"Critical error in {disease}: {str(e)}")
                continue

        print("\nTraining pipeline completed!")
        return models


if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        models = pipeline.train_all()
        print("\nTrained models:", list(models.keys()))
    except Exception as e:
        print(f"Fatal error in training pipeline: {str(e)}")