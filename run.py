import os
import sys
from pathlib import Path
from src.pipeline.training_pipeline import TrainingPipeline
from src.deployment.predict_disease import DiseasePredictor


def main():
    print("""\n
        ██████╗ ██████╗ ███████╗██████╗ ██╗ ██████╗ █████╗ ██████╗ ███████╗
        ██╔══██╗██╔══██╗██╔════╝██╔══██╗██║██╔════╝██╔══██╗██╔══██╗██╔════╝
        ██████╔╝██████╔╝█████╗  ██║  ██║██║██║     ███████║██████╔╝█████╗  
        ██╔═══╝ ██╔══██╗██╔══╝  ██║  ██║██║██║     ██╔══██║██╔══██╗██╔══╝  
        ██║     ██║  ██║███████╗██████╔╝██║╚██████╗██║  ██║██║  ██║███████╗
        ╚═╝     ╚═╝  ╚═╝╚══════╝╚═════╝ ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝
        """)
    # 1. Train all models
    print("=== TRAINING PHASE ===")
    TrainingPipeline().train_all()

    # 2. Run sample predictions
    print("\n=== TESTING PHASE ===")
    test_files = {
        'heart': 'test_samples/heart_test.csv',
        'diabetes': 'test_samples/diabetes_test.csv',
        'cancer': 'test_samples/cancer_test.csv'
    }

    predictor = DiseasePredictor()
    for disease, file in test_files.items():
        try:
            print(f"\nTesting {disease}...")
            result = predictor.predict(file)
            print(f"Result: {result['disease']}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.2%}")
        except Exception as e:
            print(f"Failed to predict {disease}: {str(e)}")


if __name__ == "__main__":
    # Add project root to Python path
    sys.path.append(str(Path(__file__).parent))
    main()