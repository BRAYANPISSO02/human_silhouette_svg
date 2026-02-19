import cv2
import numpy as np

def segment_foreground_grabcut(image: np.ndarray) -> np.ndarray:
    """
    Segment foreground using GrabCut.
    """
    mask = np.zeros(image.shape[:2], np.uint8)

    height, width = image.shape[:2]
    rect = (
        int(width * 0.1),
        int(height * 0.05),
        int(width * 0.8),
        int(height * 0.9)
    )

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        image,
        mask,
        rect,
        bgd_model,
        fgd_model,
        5,
        cv2.GC_INIT_WITH_RECT
    )

    binary_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype("uint8")

    return binary_mask