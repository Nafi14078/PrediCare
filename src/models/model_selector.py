import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_and_select_model(X, y):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, random_state=42))
        ]),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=42))
        ]),
        "RandomForest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
    }

    best_score = 0
    best_model = None
    best_model_name = ""

    for name, model in models.items():
        print(f"[INFO] Training model: {name}")
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        mean_score = np.mean(scores)
        print(f"[INFO] {name} CV Accuracy: {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_model = model
            best_model_name = name

    print(f"[INFO] Best model: {best_model_name} (Accuracy: {best_score:.4f})")
    best_model.fit(X, y)
    return best_model, best_model_name
