import tensorflow as tf
import numpy as np

def augment_image(img):
    # Applies random augmentation: flip, rotation, zoom, brightness
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.rot90(img, k=np.random.randint(0, 4))
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.image.random_zoom(img, (0.9, 1.1)) if hasattr(tf.image, 'random_zoom') else img
    return img

def augment_batch(batch):
    return tf.stack([augment_image(im) for im in batch])

# Example usage in preprocessing script
if __name__ == "__main__":
    import cv2
    import os

    input_dir = "data/raw"
    output_dir = "data/processed"

    img_size = 256
    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)
        save_dir = os.path.join(output_dir, class_name)
        os.makedirs(save_dir, exist_ok=True)

        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_size, img_size))
            img = img / 255.0
            img_tf = tf.convert_to_tensor(img, dtype=tf.float32)
            aug_img = augment_image(img_tf)
            aug_img_np = (aug_img.numpy() * 255).astype(np.uint8)
            save_path = os.path.join(save_dir, f"aug_{img_file}")
            cv2.imwrite(save_path, aug_img_np)
