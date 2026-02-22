import cv2
import numpy as np

def _resize_with_padding(img, size):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def _normalize(img):
    return img.astype(np.float32) / 255.0

def _enhance_contrast(img):
    img = (img * 255).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

#------------------------------------------------------------------------------------------------------

def _preprocess_base(image: np.ndarray, size: int) -> np.ndarray:
    """
    Preprocessing común:
    - resize + padding
    - mejora de contraste
    Retorna uint8 [0–255]
    """
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = _resize_with_padding(image, size)
    return image

def preprocess_for_segmentation(image: np.ndarray, size: int = 512) -> np.ndarray:
    return _preprocess_base(image, size)

def preprocess_for_ml(image: np.ndarray, size: int = 512) -> np.ndarray:
    image = _preprocess_base(image, size)
    image = _enhance_contrast(image)
    image = _normalize(image)
    return image