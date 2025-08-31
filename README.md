# Brain Tumor Detection Using Deep Learning
# Overview
This project detects and classifies brain tumors in MRI images using deep learning and exploratory data analysis (EDA). The workflow covers data preprocessing, visualization, training, validation, and testing. All key results and visualizations are saved in the results/ folder for transparency and reproducibility.

# Table of Contents
Project Structure

Dataset Details

EDA Highlights

Model Architecture & Training

Results

Usage Guide

Dependencies

License

# Project Structure
PrediCare/
├── data/
│   ├── processed/
│   └── raw/
│       ├── Training/
│       │   ├── glioma/
│       │   ├── meningioma/
│       │   ├── notumor/
│       │   └── pituitary/
│       └── Testing/
│           ├── glioma/
│           ├── meningioma/
│           ├── notumor/
│           └── pituitary/
├── notebooks/
│   ├── Brain_Tumor_EDA.ipynb
│   └── results/
│       ├── boxplot_height_by_class.png
│       ├── boxplot_width_by_class.png
│       ├── class_distribution.png
│       ├── correlation_heatmap.png
│       ├── pixel_intensity_distribution.png
│       └── sample_images.png
├── results/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── cross-validation report.txt
│   └── roc_curve.png
├── saved_models/
│   ├── model_fold_1.h5
│   ├── model_fold_2.h5
│   ├── model_fold_3.h5
│   ├── model_fold_4.h5
│   └── model_fold_5.h5
├── src/
│   ├── cross_validation/
│   │   ├── __init__.py
│   │   └── cv_utils.py
│   ├── deployment/
│   │   ├── __init__.py
│   │   └── inference.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── cnn_model.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── evaluate_and_ensemble.py
│   │   └── train_pipeline.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── augment.py
│   │   ├── preprocess.py
│── check_datasets.py
│── main.py
│── test_samples/
├── venv/
└── .gitignore
|__requirement.txt
|__ readme.md


# Dataset Details
Training Images: 5,712

Testing Images: 1,311

Classes: glioma, meningioma, notumor, pituitary

# Class Distribution:
glioma: 1,321 images (train), 300 images (test)

meningioma: 1,339 images (train), 306 images (test)

notumor: 1,595 images (train), 405 images (test)

pituitary: 1,457 images (train), 300 images (test)

# EDA Highlights

All exploratory graphs are in results/:

Class Distribution Barplot

Sample Images From Each Class

Pixel Intensity Histograms

Boxplots of Image Width & Height by Class

Correlation Heatmap of Image Dimensions

EDA helps reveal data imbalance, common image sizes, and pixel distribution patterns, ensuring better model preparation.


# Model Architecture & Training

Model Type: Convolutional Neural Network ( CNN)

Training: 5-fold cross-validation with stratified splits

Validation: Per-fold accuracy recorded, mean accuracy reported

Testing: Independent hold-out set (never seen in training)

# Preprocessing:

Brain region cropping

Resizing (e.g., 240x240, RGB)

Pixel normalization

Data augmentation (rotations, flips, zoom, etc.)

# Results

Cross-Validation Accuracy:

Fold 1: 0.8863

Fold 2: 0.8968

Fold 3: 0.8783

Fold 4: 0.8888

Fold 5: 0.8958

Mean accuracy: 0.8892

Testing results and confusion matrix shown in /results/cross_validation_report.txt.

Testing accuracy:0.87

# Usage Guide

Clone Repository

Install Requirements

pip install -r requirements.txt

Prepare Data

Place images in data/raw/Training and data/raw/Testing folders by class.

Run EDA Notebook

Open Brain_Tumor_EDA.ipynb in Jupyter.

Run all cells to generate visualizations.

Train Model

Use training_pipeline.py for K-fold cross-validation.

Results saved automatically in results/.

Evaluate on Test Set

Evaluate trained model(s) on testing images.

# Dependencies
Python ≥3.7

matplotlib

seaborn

pandas

numpy

Pillow

scikit-learn

tensorflow / pytorch (choose one)





