# src/pipeline/training_pipeline.py

from src.preprocessing.clean_data import clean_all_data
from src.cross_validation.train_test_split import split_data
from src.preprocessing.feature_engineering import scale_features
from src.models.model_selector import train_and_evaluate
from src.evaluation.evaluate_metrics import print_classification_report

def run_training(test_size=0.2, random_state=42):
    print("Loading and cleaning data...")
    heart, diabetes, breast_cancer = clean_all_data()

    datasets = {
        'Heart Disease': (heart.drop('target', axis=1), heart['target']),
        'Diabetes': (diabetes.drop('Outcome', axis=1), diabetes['Outcome']),
        'Breast Cancer': (breast_cancer.drop('diagnosis', axis=1), breast_cancer['diagnosis'])
    }

    for name, (X, y) in datasets.items():
        print(f"\nTraining on {name} dataset...")
        # Split data with stratification
        X_train, X_val, y_train, y_val = split_data(X, y, test_size=test_size, random_state=random_state, stratify=True)

        # Scale features
        X_train_scaled, X_val_scaled, scaler = scale_features(X_train, X_val)

        # Train models and select best
        best_model, results = train_and_evaluate(X_train_scaled, y_train, X_val_scaled, y_val)

        # Evaluation report
        print_classification_report(best_model, X_val_scaled, y_val)

if __name__ == "__main__":
    run_training()
