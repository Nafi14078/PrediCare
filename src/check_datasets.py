import os
import pandas as pd
from configparser import ConfigParser
import yaml


def check_datasets(config_path='config.yaml'):
    """Check if all datasets exist and are valid"""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    missing_files = []
    invalid_files = []

    for disease, path in config['data'].items():
        if not os.path.exists(path):
            missing_files.append(path)
            continue

        try:
            df = pd.read_csv(path)
            if df.empty:
                invalid_files.append(path)
        except:
            invalid_files.append(path)

    if missing_files:
        raise FileNotFoundError(f"Missing files: {missing_files}")
    if invalid_files:
        raise ValueError(f"Invalid files: {invalid_files}")

    print("All datasets are valid and present")
    return True