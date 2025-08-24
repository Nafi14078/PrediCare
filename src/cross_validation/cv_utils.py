import numpy as np
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from src.models.cnn_model import build_cnn


def run_cross_validation(X, y, num_classes, img_size, k=5, epochs=10, batch_size=32, lr=0.0001):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n[Fold {fold + 1}] Training...")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # One-hot encoding
        y_train_cat = to_categorical(y_train, num_classes)
        y_val_cat = to_categorical(y_val, num_classes)

        model = build_cnn((img_size, img_size, 3), num_classes)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(X_train, y_train_cat, validation_data=(X_val, y_val_cat),
                            epochs=epochs, batch_size=batch_size, verbose=2)

        val_acc = history.history['val_accuracy'][-1]
        fold_accuracies.append(val_acc)
        print(f"[Fold {fold + 1}] Validation Accuracy: {val_acc:.4f}")

    print("\nCross-validation results:")
    for idx, acc in enumerate(fold_accuracies):
        print(f"  Fold {idx + 1}: {acc:.4f}")
    print(f"Mean accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")

# Example usage (add this to your main pipeline after loading data):
# run_cross_validation(X, y, num_classes=4, img_size=256, k=5, epochs=10, batch_size=32, lr=0.0001)
