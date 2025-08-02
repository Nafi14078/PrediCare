# cleanup_models.py
import os
import glob


def cleanup():
    patterns = [
        '*_model_*.pkl',  # Remove versioned files
        '*_scaler.pkl'  # Remove old scalers
    ]

    for pattern in patterns:
        for f in glob.glob(f'saved_models/{pattern}'):
            try:
                os.remove(f)
                print(f"Removed: {f}")
            except Exception as e:
                print(f"Error removing {f}: {str(e)}")


if __name__ == "__main__":
    print("Cleaning up old model files...")
    cleanup()