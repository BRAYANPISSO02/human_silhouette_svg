import cv2
import numpy as np

def clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    Apply morphological operations to clean the mask.
    """
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask