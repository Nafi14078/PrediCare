import os


def check_dataset_integrity(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

    subfolders = os.listdir(dataset_path)
    if len(subfolders) == 0:
        raise ValueError("Dataset directory is empty.")

    for folder in subfolders:
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue
        files = os.listdir(folder_path)
        if len(files) == 0:
            raise ValueError(f"Class folder '{folder}' is empty.")
        print(f"Class '{folder}': {len(files)} images found.")

    print("Dataset sanity check passed.")


if __name__ == "__main__":
    dataset_path = "data/raw"
    check_dataset_integrity(dataset_path)
