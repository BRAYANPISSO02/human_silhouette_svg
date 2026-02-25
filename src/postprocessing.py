import cv2
import numpy as np


def clean_mask(mask: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """
    Cleans a binary mask using morphological operations.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask
