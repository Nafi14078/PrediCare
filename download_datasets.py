import os
import pandas as pd

os.makedirs('data/raw', exist_ok=True)

# Heart Disease
heart_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
heart_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
              'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

heart_df = pd.read_csv(heart_url, header=None, names=heart_cols)
heart_df.replace('?', pd.NA, inplace=True)
heart_df.dropna(inplace=True)
heart_df['ca'] = heart_df['ca'].astype(float).astype(int)
heart_df['thal'] = heart_df['thal'].astype(float).astype(int)
heart_df['target'] = heart_df['target'].apply(lambda x: 1 if x > 0 else 0)

heart_df.to_csv('data/raw/heart_disease.csv', index=False)
print("Saved heart_disease.csv")

# Diabetes
diabetes_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
diabetes_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

diabetes_df = pd.read_csv(diabetes_url, header=None, names=diabetes_cols)
diabetes_df.to_csv('data/raw/diabetes.csv', index=False)
print("Saved diabetes.csv")

# Breast Cancer
breast_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
breast_cols = ['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
               'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
               'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
               'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
               'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se',
               'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
               'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
               'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']

breast_df = pd.read_csv(breast_url, header=None, names=breast_cols)
breast_df.to_csv('data/raw/breast_cancer.csv', index=False)
print("Saved breast_cancer.csv")
