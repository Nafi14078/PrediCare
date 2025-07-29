# src/cross_validation/train_test_split.py

from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.2, random_state=42, stratify=True):
    stratify_param = y if stratify else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify_param)

if __name__ == "__main__":
    print("Use split_data() to split your dataset.")
