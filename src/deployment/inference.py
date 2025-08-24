import tensorflow as tf
import numpy as np
from src.preprocessing.preprocess import preprocess_image
import yaml

def predict(image_path):
    # Load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    model_path = config["model_save_path"]
    img_size = config["img_size"]
    num_classes = config["num_classes"]

    # Load model
    model = tf.keras.models.load_model(model_path)

    # Preprocess image
    img = preprocess_image(image_path, img_size)
    img = np.expand_dims(img, axis=0)  # batch dimension

    # Predict
    predictions = model.predict(img)
    pred_class = np.argmax(predictions, axis=1)[0]

    return pred_class, predictions

if __name__ == "__main__":
    import sys
    image_path = sys.argv[1]
    pred_class, pred_probs = predict(image_path)
    print(f"Predicted class index: {pred_class}")
    print(f"Class probabilities: {pred_probs}")
