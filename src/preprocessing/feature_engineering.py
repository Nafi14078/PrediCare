# src/preprocessing/feature_engineering.py

from sklearn.preprocessing import StandardScaler


def scale_features(X_train, X_val, X_test=None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler

    return X_train_scaled, X_val_scaled, scaler


if __name__ == "__main__":
    print("Add feature engineering functions here.")
