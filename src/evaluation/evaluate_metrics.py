# src/evaluation/evaluate_metrics.py

from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class Evaluator:
    @staticmethod
    def calculate_metrics(y_true, y_pred, average='weighted'):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average),
            'recall': recall_score(y_true, y_pred, average=average),
            'f1': f1_score(y_true, y_pred, average=average)
        }

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes=None):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

    @staticmethod
    def full_report(y_true, y_pred, classes=None):
        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=classes))

        print("\nMetrics:")
        metrics = Evaluator.calculate_metrics(y_true, y_pred)
        for name, value in metrics.items():
            print(f"{name.capitalize()}: {value:.4f}")

        if classes:
            Evaluator.plot_confusion_matrix(y_true, y_pred, classes)