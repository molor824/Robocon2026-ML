import cv2 as cv
import numpy as np
import os

def add_gaussian_noise(image: cv.Mat, mean=0, std=15):
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)

def add_contrast(image: cv.Mat, factor=1.0, channel_shift=(0.0, 0.0, 0.0)):
    img = image.astype(np.float32) * factor
    img += np.array(channel_shift, dtype=np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)

def add_blur(image: cv.Mat, ksize=3):
    return cv.GaussianBlur(image, (ksize, ksize), 0)

DATASET_PATH = "../Robocon2026Simulation/.datasets"

SRC_IMG_DIRS = [os.path.join(DATASET_PATH, "images/train"), os.path.join(DATASET_PATH, "images/val")]
SRC_LABEL_DIRS = [path.replace("images", "labels") for path in SRC_IMG_DIRS]

SRC_IMG_PATHS = [os.path.join(directory, path) for directory in SRC_IMG_DIRS for path in os.listdir(directory)]
SRC_LABEL_PATHS = [os.path.join(directory, path) for directory in SRC_LABEL_DIRS for path in os.listdir(directory)]

def main():
    for path in SRC_IMG_DIRS:
        os.makedirs(path.replace("images", "augmented/images"), exist_ok=True)
    for path in SRC_LABEL_DIRS:
        os.makedirs(path.replace("labels", "augmented/labels"), exist_ok=True)
    
    save = False
    for path in SRC_IMG_PATHS:
        img = cv.imread(path)
        dst_path = path.replace("images", "augmented/images")
        augmented = add_contrast(img, np.random.randn() * 0.1 + 1, np.random.randn(3) * (5, 10, 5))
        augmented = add_gaussian_noise(augmented, np.random.randn() * 0.1, max(0.0, np.random.randn() * 5 + 10))
        augmented = add_blur(augmented, np.random.randint(0, 3) * 2 + 1)
        cv.imshow("Image", img)
        cv.imshow("Augmented", augmented)

        cv.imwrite(dst_path, img)

        key = cv.waitKey(1 if save else 0)
        if key == ord('q'):
            break
        elif key == ord('s'): # Save all image
            save = True

if __name__ == "__main__":
    main()
