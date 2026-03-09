import os
import sys
from skimage import io
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms  # , utils

# import torch.optim as optim

import numpy as np
from PIL import Image
import glob


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


def save_output(image_name, pred, d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert("RGB")
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    full_path = os.path.join(d_dir, imidx + ".png")
    imo.save(full_path)


def get_binary_mask(pred, original_shape, threshold=0.5):
    """
    Convierte la predicción en un array binario de NumPy (0 y 1)
    con las dimensiones originales de la imagen.
    """
    # 1. Squeeze para quitar dimensiones extra (batch, canal)
    predict = pred.squeeze().cpu().data.numpy()

    # 2. Convertir a imagen de PIL para redimensionar con calidad
    # (Usamos el tamaño original: original_shape es (H, W))
    predict_img = Image.fromarray(predict * 255).convert("L")
    predict_rescaled = predict_img.resize(
        (original_shape[1], original_shape[0]), resample=Image.BILINEAR
    )

    # 3. Volver a numpy y binarizar
    mask_np = np.array(predict_rescaled) / 255.0  # Volver a rango [0, 1]
    binary_mask = (mask_np > threshold).astype(
        np.uint8
    )  # 1 donde hay objeto, 0 donde no

    return binary_mask
