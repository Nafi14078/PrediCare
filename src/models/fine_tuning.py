from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import numpy as np
import joblib


class ModelTuner:
    def __init__(self, model_type='random_forest', search_type='grid', cv=5):
        self.model_type = model_type
        self.search_type = search_type
        self.cv = cv
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None

        self._setup_models()
        self._setup_param_grids()

    def _setup_models(self):
        self.models = {
            'random_forest': RandomForestClassifier(),
            'xgboost': XGBClassifier(),
            'svm': SVC(probability=True)
        }

    def _setup_param_grids(self):
        self.param_grids = {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.1, 1],
                'kernel': ['linear', 'rbf', 'poly']
            }
        }

    def tune(self, X, y, n_iter=50):
        """Perform hyperparameter tuning"""
        param_grid = self.param_grids[self.model_type]
        model = self.models[self.model_type]

        if self.search_type == 'grid':
            search = GridSearchCV(model, param_grid, cv=self.cv,
                                  scoring='accuracy', n_jobs=-1)
        else:
            search = RandomizedSearchCV(model, param_grid, cv=self.cv,
                                        n_iter=n_iter, scoring='accuracy',
                                        random_state=42, n_jobs=-1)

        search.fit(X, y)

        self.best_params_ = search.best_params_
        self.best_score_ = search.best_score_
        self.best_estimator_ = search.best_estimator_

        print(f"Best parameters: {self.best_params_}")
        print(f"Best accuracy: {self.best_score_:.4f}")

        return self.best_estimator_

    def save_best_model(self, filepath):
        """Save the best tuned model"""
        if self.best_estimator_ is None:
            raise ValueError("No model has been tuned yet")
        joblib.dump(self.best_estimator_, filepath)
        print(f"Best model saved to {filepath}")

# Example usage:
# tuner = ModelTuner(model_type='random_forest', search_type='random')
# best_model = tuner.tune(X_train, y_train)
# tuner.save_best_model('best_rf_model.pkl')