import cv2
import numpy as np


def _resize_with_padding(img, size):
    h, w = img.shape[:2]
    # Calculamos el factor de escala manteniendo la relación de aspecto
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)

    # Cambiamos INTER_AREA por INTER_LINEAR o INTER_CUBIC si la imagen es pequeña y vamos a agrandarla.
    # INTER_AREA es mejor para achicar, INTER_CUBIC es mejor para agrandar.
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC

    resized = cv2.resize(img, (nw, nh), interpolation=interpolation)

    # Crear el lienzo (canvas) negro del tamaño deseado (1024x1024)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)

    # Calcular coordenadas para centrar la imagen
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2

    # Insertar la imagen redimensionada en el centro del canvas
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _normalize(img):
    # Asegura que el rango sea [0.0, 1.0] en float32
    return img.astype(np.float32) / 255.0


def _enhance_contrast(img):
    # Aseguramos que entre como uint8 para la conversión de color
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Ajustamos el clipLimit a 2.0 para que no genere demasiado ruido en 1024px
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ------------------------------------------------------------------------------------------------------


def _preprocess_base(image: np.ndarray, size: int) -> np.ndarray:
    """
    Preprocessing común:
    - Suavizado leve para reducir ruido antes de escalar
    - Resize con padding al nuevo tamaño (1024)
    """
    # Un kernel de 3x3 es ideal para 1024px, no borra demasiados detalles
    # image = cv2.GaussianBlur(image, (3, 3), 0)
    image = _resize_with_padding(image, size)
    return image


def preprocess_for_segmentation(image: np.ndarray, size: int = 1024) -> np.ndarray:
    """Retorna imagen uint8 [0-255] lista para el modelo de segmentación"""
    return _preprocess_base(image, size)


def preprocess_for_ml(image: np.ndarray, size: int = 1024) -> np.ndarray:
    """Retorna imagen normalizada [0.0 - 1.0] con contraste mejorado"""
    image = _preprocess_base(image, size)
    image = _enhance_contrast(image)
    image = _normalize(image)
    return image
