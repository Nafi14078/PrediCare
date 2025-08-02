from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
import joblib
import yaml
import datetime
import numpy as np
from collections import Counter


class ModelSelector:
    def __init__(self, config_path='config.yaml'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Initialize models with balanced class weights
        self.models = self._initialize_models()

    def _initialize_models(self):
        """Initialize models with class balancing and optimal defaults"""
        return {
            'random_forest': RandomForestClassifier(
                n_estimators=100,  # Reduced from default
                max_depth=5,  # Added depth limit
                min_samples_split=5,  # Increased from default 2
                class_weight='balanced',
                random_state=42
            ),
            'xgboost': XGBClassifier(
                max_depth=3,  # Shallower trees
                learning_rate=0.1,  # Lower learning rate
                subsample=0.8,  # Stochastic sampling
                colsample_bytree=0.8,  # Feature subsampling
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=1.0,  # L2 regularization
                eval_metric='logloss'
            ),
            'svm': SVC(
                C=1.0,  # Inverse regularization strength
                kernel='rbf',
                gamma='scale',  # Automatic kernel coefficient
                probability=True,
                class_weight='balanced'
            ),
            'logistic_regression': LogisticRegression(
                penalty='l2',  # Ridge regularization
                C=1.0,  # Inverse of regularization strength
                solver='liblinear',
                max_iter=1000,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=42
            )
        }

    def _calculate_scale_pos_weight(self):
        """Calculate weights for XGBoost based on class imbalance"""
        # Will be set during fit()
        return None

    def select_best_model(self, X, y, cv=5):
        """Select best model using stratified cross-validation"""
        # Update class weights based on actual data
        self.models['xgboost'].scale_pos_weight = self._get_class_ratio(y)

        best_score = -1
        best_model = None
        best_name = ""
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        for name, model in self.models.items():
            try:
                # Use F1-score for imbalanced data
                scores = cross_val_score(
                    model, X, y, cv=cv,
                    scoring='f1_weighted',
                    n_jobs=-1
                )
                mean_score = scores.mean()

                if mean_score > best_score:
                    best_score = mean_score
                    best_model = model
                    best_name = name

                print(f"{name:>20} | F1: {mean_score:.4f} | Best: {best_name}")
            except Exception as e:
                print(f"Error evaluating {name}: {str(e)}")
                continue

        print(f"\nBest model: {best_name} with weighted F1: {best_score:.4f}")

        # Fit best model with full training data
        best_model.fit(X, y)
        self._evaluate_model(best_model, X, y)

        return best_model

    def _get_class_ratio(self, y):
        """Calculate class imbalance ratio for XGBoost"""
        class_counts = Counter(y)
        return class_counts[0] / class_counts[1] if 1 in class_counts else 1.0

    def _evaluate_model(self, model, X, y):
        """Print detailed evaluation metrics"""
        y_pred = model.predict(X)
        print("\nModel Evaluation:")
        print(f"Precision: {precision_score(y, y_pred, average='weighted'):.4f}")
        print(f"Recall:    {recall_score(y, y_pred, average='weighted'):.4f}")
        print(f"F1-score:  {f1_score(y, y_pred, average='weighted'):.4f}")

        # For binary classification
        if len(np.unique(y)) == 2:
            from sklearn.metrics import roc_auc_score
            print(f"AUC-ROC:   {roc_auc_score(y, y_pred):.4f}")

    def save_model(self, model, disease_type):
        """Save model with versioning"""
        import datetime
        version = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        save_path = (
            f"{self.config['models']['save_path']}"
            f"{disease_type}_model_{version}.pkl"
        )
        joblib.dump(model, save_path)
        print(f"Model saved to {save_path}")
        return save_path

    def save_metadata(self, model, X, y, disease_type):
        """Save training metadata for reproducibility"""
        metadata = {
            'feature_names': list(X.columns) if hasattr(X, 'columns') else [],
            'class_distribution': dict(Counter(y)),
            'training_date': datetime.datetime.now().strftime("%Y-%m-%d"),
            'best_params': model.get_params() if hasattr(model, 'get_params') else {}
        }
        path = self.save_model(model, disease_type).replace('.pkl', '_meta.json')
        import json
        with open(path, 'w') as f:
            json.dump(metadata, f)