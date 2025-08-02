from src.deployment.predict_disease import DiseasePredictor

predictor = DiseasePredictor()
try:
    result = predictor.predict("test_samples/heart_test.csv")
    print("Prediction successful!")
    print(result)
except Exception as e:
    print(f"Prediction failed: {str(e)}")