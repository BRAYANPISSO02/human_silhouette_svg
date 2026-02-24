import cv2
import numpy as np


class HOGPersonDetector:
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(
            cv2.HOGDescriptor_getDefaultPeopleDetector()
        )

    def detect(self, image):
        """
        Detecta personas y retorna bounding boxes con su confidence score.
        """
        boxes, weights = self.hog.detectMultiScale(
            image,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05
        )

        # weights viene como array Nx1 → lo aplanamos
        weights = weights.flatten() if len(weights) > 0 else []

        return boxes, weights


def select_best_box(boxes, weights, image_shape,
                    min_confidence=1.0,
                    min_area_ratio=0.05):
    """
    Selecciona el mejor bounding box basado en:
    - confidence mínima
    - área mínima relativa a la imagen
    - mayor score final
    """
    if len(boxes) == 0:
        return None

    H, W = image_shape[:2]
    image_area = H * W

    candidates = []

    for (box, score) in zip(boxes, weights):
        x, y, w, h = box
        area = w * h

        if score < min_confidence:
            continue

        if area < min_area_ratio * image_area:
            continue

        # score ponderado (puedes ajustar esto si quieres)
        final_score = score * area

        candidates.append((final_score, box))

    if not candidates:
        return None

    # Elegimos el de mayor score final
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def expand_box(box, image_shape, padding_ratio=0.15):
    """
    Expande el bounding box para no cortar extremidades.
    """
    x, y, w, h = box
    H, W = image_shape[:2]

    pad_w = int(w * padding_ratio)
    pad_h = int(h * padding_ratio)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(W, x + w + pad_w)
    y2 = min(H, y + h + pad_h)

    return x1, y1, x2, y2


def reframe_to_canvas(image, roi_coords):
    """
    Coloca el ROI centrado en un canvas del mismo tamaño que la imagen original.
    """
    H, W = image.shape[:2]
    x1, y1, x2, y2 = roi_coords

    roi = image[y1:y2, x1:x2]
    h, w = roi.shape[:2]

    canvas = np.zeros_like(image)

    cx = W // 2
    cy = H // 2

    x_start = max(0, cx - w // 2)
    y_start = max(0, cy - h // 2)

    x_end = min(W, x_start + w)
    y_end = min(H, y_start + h)

    roi_x_end = x_end - x_start
    roi_y_end = y_end - y_start

    canvas[y_start:y_end, x_start:x_end] = roi[:roi_y_end, :roi_x_end]

    return canvas


def detect_and_reframe_person(image):
    """
    Detecta una persona dominante con HOG y la reencuadra.
    Si no hay detección confiable → devuelve la imagen original.
    """
    detector = HOGPersonDetector()
    boxes, weights = detector.detect(image)

    best_box = select_best_box(
        boxes,
        weights,
        image.shape,
        min_confidence=0.9,   # ajustable
        min_area_ratio=0.05   # descarta personas muy pequeñas
    )

    if best_box is None:
        # Fallback explícito y correcto
        return image

    expanded_box = expand_box(best_box, image.shape)

    reframed = reframe_to_canvas(image, expanded_box)

    return reframed