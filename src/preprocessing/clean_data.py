# src/preprocessing/clean_data.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os

# Define dataset configurations
DATASETS = {
    'heart': {
        'columns': [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
            'ca', 'thal', 'target'
        ],
        'target': 'target',
        'header': None  # No header row
    },
    'diabetes': {
        'columns': [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
        ],
        'target': 'Outcome',
        'header': 0  # Use first row as header
    },
    'cancer': {
        'columns': ['id', 'diagnosis'] + [f'feature_{i}' for i in range(1, 31)],
        'target': 'diagnosis',
        'header': None  # No header row
    }
}


def load_data(filepath):
    """Load dataset with proper configuration"""
    filename = os.path.basename(filepath)

    for name, config in DATASETS.items():
        if name in filename:
            df = pd.read_csv(
                filepath,
                header=config['header'],
                names=config['columns'] if config['header'] is None else None
            )

            # Special handling for breast cancer
            if name == 'cancer':
                df = df.drop('id', axis=1)
                df['target'] = df['diagnosis'].map({'M': 1, 'B': 0})
                df = df.drop('diagnosis', axis=1)
                config['target'] = 'target'

            # Store target column in dataframe attributes
            df.attrs['target_column'] = config['target']
            return df

    raise ValueError(f"Unknown dataset: {filename}")


def preprocess_data(df, target_col):
    """Clean and preprocess the data"""
    # Handle missing values
    df = df.replace('?', np.nan)

    # Convert to numeric (except target if it's already categorical)
    for col in df.columns:
        if col != target_col and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    features = df.drop(target_col, axis=1)
    features = pd.DataFrame(imputer.fit_transform(features),
                            columns=features.columns)

    # Scale features
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(scaler.fit_transform(features),
                                   columns=features.columns)

    # Process target (convert to binary if needed)
    if df[target_col].dtype == object:
        target = pd.factorize(df[target_col])[0]  # Convert strings to numbers
    else:
        target = df[target_col].astype(int)

    return features_scaled, target


def process_and_save(input_path, output_path):
    """Complete processing pipeline"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = load_data(input_path)
    target_col = df.attrs['target_column']

    # Special case: diabetes target is in header row
    if 'diabetes' in input_path and df[target_col].dtype == object:
        df = df[df[target_col] != target_col]  # Remove header row if it exists

    X, y = preprocess_data(df, target_col)

    # Save processed data
    pd.concat([X, y.rename('target')], axis=1).to_csv(output_path, index=False)
    print(f"\nSuccessfully processed {os.path.basename(input_path)}")
    print(f"Target distribution:\n{y.value_counts()}")
    return output_path


if __name__ == "__main__":
    datasets = {
        'heart': ('data/raw/heart_disease.csv', 'data/processed/heart_processed.csv'),
        'diabetes': ('data/raw/diabetes.csv', 'data/processed/diabetes_processed.csv'),
        'cancer': ('data/raw/breast_cancer.csv', 'data/processed/cancer_processed.csv')
    }

    for name, (input_path, output_path) in datasets.items():
        try:
            print(f"\n{'=' * 40}\nProcessing {name} dataset...")
            process_and_save(input_path, output_path)
            print(f"Saved to: {output_path}")
            print(f"Sample data:\n{pd.read_csv(output_path).head(2).to_string()}")
        except Exception as e:
            print(f"Error processing {name}: {str(e)}")