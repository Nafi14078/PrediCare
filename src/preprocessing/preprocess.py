import cv2
import numpy as np

def preprocess_image(img_path, img_size):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image at path {img_path} not found or unable to read.")
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0  # Normalize to [0,1]
    return img.astype(np.float32)
