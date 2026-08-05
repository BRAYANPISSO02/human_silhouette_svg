import numpy as np
import torch
import cv2

ADAPTATIVE_BLOCK_SIZE = 31
ADAPTATIVE_C = -5
MIN_AREA = 25
MORPH_KERNEL_SIZE = (3, 3)

def postprocess(prediction: torch.Tensor) -> np.ndarray:
    """
    Converts the raw U-Net prediction into a cleaned binary image suitable for
    SVG vectorization.
    The pipeline includes normalization, contrast stretching, adaptive
    thresholding, connected-component filtering, morphological refinement, and
    optional skeletonization.
    Args:
        prediction (torch.Tensor):
            Raw output tensor from the U-Net.
    Returns:
        numpy.ndarray:
            Binary image with white foreground and black background.
    """
        # --------------------------------------------------
    # Tensor -> NumPy
    gray = prediction.squeeze().detach().cpu().numpy()

    # [-1,1] -> [0,255]
    gray = ((gray + 1.0) / 2.0) * 255.0
    gray = np.clip(gray, 0, 255).astype(np.uint8)

    # Estiramiento automático de contraste
    p1 = np.percentile(gray, 1)
    p99 = np.percentile(gray, 99)
    gray = gray.astype(np.float32)
    gray = (gray - p1) / (p99 - p1 + 1e-8)
    gray = np.clip(gray, 0, 1)
    gray = (gray * 255).astype(np.uint8)

    # Binarización adaptativa
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        ADAPTATIVE_BLOCK_SIZE,
        ADAPTATIVE_C)

    # Eliminar componentes pequeñas
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8)
    clean = np.zeros_like(binary)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= MIN_AREA:
            clean[labels == i] = 255

    # Conectar líneas fragmentadas
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        MORPH_KERNEL_SIZE)
    clean = cv2.morphologyEx(
        clean,
        cv2.MORPH_CLOSE,
        kernel)    

    # Asegurar fondo negro y líneas blancas
    if np.mean(clean) > 127:
        clean = 255 - clean

    # Adelgazamiento (Skeleton)
    if hasattr(cv2, "ximgproc"):
        clean = cv2.ximgproc.thinning(clean)

    # Si ximgproc no está disponible, se devuelve la imagen limpia
    return clean