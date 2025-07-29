# src/models/model_selector.py

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score

def get_models():
    return {
        'SVM': SVC(probability=True, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

def train_and_evaluate(X_train, y_train, X_val, y_val):
    models = get_models()
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        results[name] = {'model': model, 'accuracy': acc}
        print(f"{name} accuracy: {acc:.4f}")
    best_model_name = max(results, key=lambda k: results[k]['accuracy'])
    print(f"Best model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.4f}")
    return results[best_model_name]['model'], results

if __name__ == "__main__":
    print("Run this module with real data and train/test splits")
