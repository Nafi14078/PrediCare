# src/preprocessing/feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, create_interactions=True, polynomial_features=False):
        self.create_interactions = create_interactions
        self.polynomial_features = polynomial_features
        self.interaction_terms = None
        self.poly = None

    def fit(self, X, y=None):
        if self.create_interactions:
            # Identify numeric columns for interactions
            numeric_cols = X.select_dtypes(include=np.number).columns
            self.interaction_terms = [(a, b) for i, a in enumerate(numeric_cols)
                                      for j, b in enumerate(numeric_cols) if i < j]

        if self.polynomial_features:
            self.poly = PolynomialFeatures(degree=2, include_bias=False)
            self.poly.fit(X[numeric_cols])

        return self

    def transform(self, X):
        X = X.copy()

        # Create interaction terms
        if self.create_interactions and self.interaction_terms:
            for a, b in self.interaction_terms:
                X[f'{a}_x_{b}'] = X[a] * X[b]

        # Create polynomial features
        if self.polynomial_features and self.poly:
            poly_features = self.poly.transform(X[self.poly.feature_names_in_])
            poly_cols = self.poly.get_feature_names_out()
            poly_df = pd.DataFrame(poly_features, columns=poly_cols)

            # Remove original columns to avoid duplication
            poly_df = poly_df.drop(columns=self.poly.feature_names_in_)
            X = pd.concat([X, poly_df], axis=1)

        # Create age bins for medical datasets
        if 'age' in X.columns:
            bins = [0, 30, 45, 60, 75, 100]
            labels = ['young', 'adult', 'middle_aged', 'senior', 'elderly']
            X['age_group'] = pd.cut(X['age'], bins=bins, labels=labels)
            X = pd.get_dummies(X, columns=['age_group'], prefix='age')

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

# Example usage:
# engineer = FeatureEngineer(create_interactions=True, polynomial_features=False)
# X_engineered = engineer.fit_transform(X)