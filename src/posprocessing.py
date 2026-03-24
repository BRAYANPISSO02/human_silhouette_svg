import cv2
import numpy as np
from skimage import io
import torch

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn

def get_binary_mask(prediction_tensor, original_img_path, threshold=0.8):
    """
    Toma la salida del modelo y devuelve una máscara binaria del tamaño original.
    """
    # 1. Normalización (asumiendo que d1 es el tensor de salida)
    # Nota: Si normPRED es una función externa, también muévela aquí
    pred = prediction_tensor[:, 0, :, :]
    
    # Aquí asumo que normPRED ya está definida en este archivo o importada
    pred = normPRED(pred) 

    # 2. Pasar a numpy
    pred_np = pred.cpu().data.numpy()[0]

    # 3. Suavizado
    pred_smooth = cv2.GaussianBlur(pred_np, (5, 5), 0)

    # 4. Normalización robusta
    pred_smooth = (pred_smooth - pred_smooth.min()) / (
        pred_smooth.max() - pred_smooth.min() + 1e-8
    )

    # 5. Obtener dimensiones originales
    img_ori = io.imread(original_img_path)
    h, w = img_ori.shape[:2]

    # 6. Redimensionar
    pred_resized = cv2.resize(pred_smooth, (w, h))

    # 7. Binarización
    mask_binaria = (pred_resized > threshold).astype(np.uint8)
    
    return mask_binaria