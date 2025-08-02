import argparse
import yaml
from src.pipeline.training_pipeline import TrainingPipeline
from src.deployment.predict_disease import DiseasePredictor


def main():
    parser = argparse.ArgumentParser(description="Predicare Disease Prediction System")
    parser.add_argument('--train', action='store_true', help='Train all models')
    parser.add_argument('--predict', action='store_true', help='Make predictions')
    parser.add_argument('--data', type=str, help='Path to data for prediction')
    args = parser.parse_args()

    if args.train:
        print("Training all models...")
        pipeline = TrainingPipeline()
        pipeline.train_all()
        print("Training completed!")

    if args.predict:
        if not args.data:
            raise ValueError("Please provide data path with --data")

        predictor = DiseasePredictor()
        result = predictor.predict(args.data)
        print("\nPrediction Result:")
        print(f"Most likely condition: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nAll probabilities:")
        for disease, prob in result['probabilities'].items():
            print(f"{disease.replace('_', ' ').title()}: {prob:.2%}")


if __name__ == "__main__":
    main()