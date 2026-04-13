import cv2 as cv
import numpy as np

def add_gaussian_noise(image: cv.Mat, mean=0, std=15):
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)

