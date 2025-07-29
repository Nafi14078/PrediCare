# src/preprocessing/clean_data.py

import pandas as pd
import numpy as np

def load_heart_disease(path):
    df = pd.read_csv(path)
    # Example cleaning steps
    df = df.replace('?', np.nan)
    df = df.dropna()
    df['target'] = df['target'].astype(int)
    return df

def load_diabetes(path):
    df = pd.read_csv(path)
    # Replace zeros in some columns that shouldn't be zero with NaN
    cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_zero:
        df[col] = df[col].replace(0, np.nan)
    df = df.fillna(df.median())
    return df

def load_breast_cancer(path):
    df = pd.read_csv(path)
    # Remove ID column if present
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    # Encode diagnosis as 0/1
    df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})
    return df

def clean_all_data(raw_data_dir='data/raw/'):
    heart = load_heart_disease(raw_data_dir + 'heart_disease.csv')
    diabetes = load_diabetes(raw_data_dir + 'diabetes.csv')
    breast_cancer = load_breast_cancer(raw_data_dir + 'breast_cancer.csv')
    return heart, diabetes, breast_cancer

if __name__ == "__main__":
    heart, diabetes, breast_cancer = clean_all_data()
    print("Heart Disease Data shape:", heart.shape)
    print("Diabetes Data shape:", diabetes.shape)
    print("Breast Cancer Data shape:", breast_cancer.shape)
