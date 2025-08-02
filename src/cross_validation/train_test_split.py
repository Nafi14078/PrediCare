# src/cross_validation/train_test_split.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Tuple, Union


class DataSplitter:
    def __init__(self, test_size: float = 0.2, random_state: int = 42,
                 stratify: bool = True, n_splits: int = 5):
        """
        Initialize the data splitter

        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            stratify: Whether to maintain class distribution in splits
            n_splits: Number of folds for cross-validation
        """
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify
        self.n_splits = n_splits

    def basic_split(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> Tuple:
        """
        Basic train-test split

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            X_train, X_test, y_train, y_test
        """
        stratify = y if self.stratify else None
        return train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify
        )

    def kfold_splits(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> list:
        """
        Generate K-fold cross-validation splits

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            List of (train_index, test_index) tuples
        """
        kf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )
        return list(kf.split(X, y))

    def time_series_split(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray],
                          date_col: str) -> Tuple:
        """
        Time-based train-test split

        Args:
            X: Feature matrix including date column
            y: Target vector
            date_col: Name of the date column

        Returns:
            X_train, X_test, y_train, y_test
        """
        if date_col not in X.columns:
            raise ValueError(f"Date column '{date_col}' not found in features")

        X_sorted = X.sort_values(date_col)
        y_sorted = y[X_sorted.index]

        split_idx = int(len(X_sorted) * (1 - self.test_size))

        X_train = X_sorted.iloc[:split_idx].drop(date_col, axis=1)
        X_test = X_sorted.iloc[split_idx:].drop(date_col, axis=1)
        y_train = y_sorted.iloc[:split_idx]
        y_test = y_sorted.iloc[split_idx:]

        return X_train, X_test, y_train, y_test

    def get_split_method(self, split_type: str = 'basic'):
        """
        Get the appropriate split method

        Args:
            split_type: Type of split ('basic', 'kfold', 'timeseries')

        Returns:
            The appropriate split method
        """
        methods = {
            'basic': self.basic_split,
            'kfold': self.kfold_splits,
            'timeseries': self.time_series_split
        }

        if split_type not in methods:
            raise ValueError(f"Unknown split type: {split_type}. Choose from {list(methods.keys())}")

        return methods[split_type]

# Example usage:
# splitter = DataSplitter(test_size=0.3, random_state=42)
# X_train, X_test, y_train, y_test = splitter.basic_split(X, y)
# Or for cross-validation:
# folds = splitter.kfold_splits(X, y)