# src/evaluation/evaluate_metrics.py

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def print_classification_report(model, X_val, y_val):
    preds = model.predict(X_val)
    print("Classification Report:")
    print(classification_report(y_val, preds))

    cm = confusion_matrix(y_val, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

if __name__ == "__main__":
    print("Run this module with a trained model and validation data")
