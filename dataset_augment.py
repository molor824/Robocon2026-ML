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
    for img_path in SRC_IMG_DIRS:
        os.makedirs(img_path.replace("images", "augmented/images"), exist_ok=True)
    for img_path in SRC_LABEL_DIRS:
        os.makedirs(img_path.replace("labels", "augmented/labels"), exist_ok=True)
    
    save = False
    try:
        for img_path, label_path in zip(SRC_IMG_PATHS, SRC_LABEL_PATHS):
            img = cv.imread(img_path)
            with open(label_path, "r") as f:
                label = f.read()
            dst_img_path = img_path.replace("images", "augmented/images")
            dst_label_path = label_path.replace("labels", "augmented/labels")
            augmented = add_contrast(img, np.random.randn() * 0.1 + 1, np.random.randn(3) * (5, 10, 5))
            augmented = add_gaussian_noise(augmented, np.random.randn() * 0.1, max(0.0, np.random.randn() * 5 + 10))
            augmented = add_blur(augmented, np.random.randint(0, 3) * 2 + 1)

            print(f"Saved to {dst_img_path}")
            with open(dst_label_path, "w") as f:
                f.write(label)
            cv.imwrite(dst_img_path, augmented)

            if not save:
                cv.imshow("Image", img)
                cv.imshow("Augmented", augmented)

                key = cv.waitKey(0)
                if key == ord('q'):
                    break
                elif key == ord('s'): # Save all image
                    save = True
    except KeyboardInterrupt:
        pass
    finally:
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
