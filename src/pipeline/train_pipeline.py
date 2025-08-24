import os
import yaml
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from src.models.cnn_model import build_cnn

def list_images_and_labels(data_dir):
    img_paths = []
    labels = []
    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            if not os.path.isfile(img_path):
                continue
            img_paths.append(img_path)
            labels.append(class_name)
    return img_paths, labels

def kfold_generator(data_dir, img_size, batch_size, num_classes, epochs, k=5):
    # List images and labels from the "Training" folder
    img_paths, labels = list_images_and_labels(os.path.join(data_dir, "Training"))
    df = pd.DataFrame({'filename': img_paths, 'class': labels})
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    y = df['class'].values

    # Data augmentation generator
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True
    )

    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df['filename'], y)):
        print(f"\n[Fold {fold + 1}] Training...")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_gen = datagen.flow_from_dataframe(
            train_df,
            x_col='filename',
            y_col='class',
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )

        val_gen = datagen.flow_from_dataframe(
            val_df,
            x_col='filename',
            y_col='class',
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )

        model = build_cnn((img_size, img_size, 3), num_classes)
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            verbose=2
        )

        val_acc = history.history['val_accuracy'][-1]
        fold_accuracies.append(val_acc)
        print(f"[Fold {fold + 1}] Validation Accuracy: {val_acc:.4f}")

        # SAVE MODEL FOR EACH FOLD
        model_save_dir = "saved_models"
        os.makedirs(model_save_dir, exist_ok=True)
        model.save(os.path.join(model_save_dir, f"model_fold_{fold+1}.h5"))
        print(f"Model for fold {fold+1} saved to {model_save_dir}/model_fold_{fold+1}.h5")

    print("\nK-fold cross-validation results:")
    for idx, acc in enumerate(fold_accuracies):
        print(f"  Fold {idx + 1}: {acc:.4f}")
    print(f"Mean accuracy: {sum(fold_accuracies)/len(fold_accuracies):.4f}")

def train():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    img_size = config["img_size"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    num_classes = config["num_classes"]
    k_folds = config.get("k_folds", 5)  # Optional; default to 5 folds

    kfold_generator("data/raw", img_size, batch_size, num_classes, epochs, k=k_folds)

if __name__ == "__main__":
    train()
