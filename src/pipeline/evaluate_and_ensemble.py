# File: src/pipeline/evaluate_and_ensemble.py

import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.evaluation.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import seaborn as sns
from sklearn.metrics import confusion_matrix

# === Configuration ===

# Paths to your saved fold models
model_paths = [
    'saved_models/model_fold_1.h5',
    'saved_models/model_fold_2.h5',
    'saved_models/model_fold_3.h5',
    'saved_models/model_fold_4.h5',
    'saved_models/model_fold_5.h5',
]

# Test data directory
test_data_dir = "D:/PrediCare/data/raw/Testing"  # TODO: Update to your test data folder path

# Image input size (update if different)
img_height, img_width = 256, 256
batch_size = 32

# === Create results folder if it doesn't exist ===

results_folder = 'results'
os.makedirs(results_folder, exist_ok=True)

# === Prepare Test Data Generator ===

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Important: do NOT shuffle for consistent order
)

# === Load all fold models ===

models = []
print("Loading fold models...")
for path in model_paths:
    print(f"Loading model from {path}")
    models.append(load_model(path))

# === Evaluate each fold model on test data ===

print("\nEvaluating each fold model:")
for i, model in enumerate(models, 1):
    loss, accuracy = model.evaluate(test_generator, verbose=1)
    print(f"Fold {i} model test accuracy: {accuracy:.4f}, loss: {loss:.4f}")


# === Ensemble prediction by averaging probabilities ===

def ensemble_predict(models, data_generator):
    """Predict by averaging outputs from all models"""
    preds = [model.predict(data_generator, verbose=0) for model in models]
    avg_preds = np.mean(preds, axis=0)
    return avg_preds


print("\nCalculating ensemble predictions...")
ensemble_preds = ensemble_predict(models, test_generator)

# Convert averaged probabilities to predicted class labels
ensemble_labels = np.argmax(ensemble_preds, axis=1)

# True labels from generator (ensure shuffle=False when creating generator)
true_labels = test_generator.classes

# Calculate ensemble accuracy
ensemble_accuracy = accuracy_score(true_labels, ensemble_labels)
print(f"Ensemble test accuracy: {ensemble_accuracy:.4f}")


# === Save classification report ===

def save_classification_report(y_true, y_pred, class_names, save_path):
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(save_path, 'w') as f:
        f.write(report)
    print(f"Classification report saved to {save_path}")


class_names = list(test_generator.class_indices.keys())

classification_report_path = os.path.join(results_folder, 'classification_report.txt')
save_classification_report(true_labels, ensemble_labels, class_names, classification_report_path)


# === Plot and save Confusion Matrix ===

def plot_and_save_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(save_path, dpi=300)
    plt.close()  # Close figure to free memory


conf_matrix_path = os.path.join(results_folder, 'confusion_matrix.png')
plot_and_save_confusion_matrix(true_labels, ensemble_labels, class_names, conf_matrix_path)
print(f"Confusion Matrix saved to {conf_matrix_path}")


# === Plot and save ROC Curve ===

def plot_and_save_roc(y_true, y_pred_proba, class_names, save_path):
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()  # Close figure to free memory


roc_curve_path = os.path.join(results_folder, 'roc_curve.png')
plot_and_save_roc(true_labels, ensemble_preds, class_names, roc_curve_path)
print(f"ROC Curve saved to {roc_curve_path}")
