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

from preprocessing import RescaleT
from preprocessing import ToTensorLab
from preprocessing import SalObjDataset
from segmentation import normPRED, get_binary_mask, save_output
from vectorization import extract_main_contour, contour_to_svg


root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)

from model_u2net import U2NET  # full size version 173.6 MB


def main():

    # PREPROCESAMIENTO
    # --------- 1. get image path and name ---------
    model_name = "u2net"

    image_dir = os.path.join(os.getcwd(), "data", "photo")
    prediction_dir = os.path.join(os.getcwd(), "outputs", "debug")
    os.makedirs(prediction_dir, exist_ok=True)
    model_dir = os.path.join(
        os.getcwd(),
        "model_u2net",
        model_name + "_human_seg",
        model_name + "_human_seg.pth",
    )

    #Agregamos todas las rutas de todos lo archivos dentro de /photo a una lista 
    img_name_list = glob.glob(os.path.join(image_dir, "*"))
    print(img_name_list)

    # -- Preparamos la "línea de producción" para la Red Neuronal --
    
    test_salobj_dataset = SalObjDataset(
        img_name_list=img_name_list,
        lbl_name_list=[],
        transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)]),
    )
    test_salobj_dataloader = DataLoader(
        test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1
    )

    # OBTENCIÓN DE SILUETA DE PERSONAS

    # --------- Cargamos el modelo ---------
    if model_name == "u2net":
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 1)

    #Si hay una tarjeta de video NVIDIA potente (GPU), pasar todo para procesar más rapido 
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else: #Sino mantengala con en la CPU (más lento pero funciona en cualquier pc) 
        net.load_state_dict(torch.load(model_dir, map_location="cpu"))
    net.eval()

    # --------- Inferencia para cada imagen ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:", img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test["image"]
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        # Se pasa por la red neuronal (el más importante de los 7 mapas es d1)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        # normalization
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        # Leemos la imagen original para saber sus dimensiones reales
        img_ori = io.imread(img_name_list[i_test])

        # Obtenemos el array de 0s y 1s
        mask_binaria = get_binary_mask(pred, img_ori.shape[:2], threshold=0.5)

        # print(
        #     f"Máscara generada. Forma: {mask_binaria.shape}, Valores únicos: {np.unique(mask_binaria)}"
        # )
        # Aquí ya puedes usar 'mask_binaria' para lo que necesites
        # -----------------------------------------

        # (Opcional) Si aún quieres guardar el archivo visual:
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        Image.fromarray(mask_binaria * 255).save(
            os.path.join(
                prediction_dir, img_name_list[i_test].split(os.sep)[-1] + ".png"
            )
        )

        mask_for_cv2 = (mask_binaria * 255).astype(np.uint8)
        h, w = mask_binaria.shape
        nombre_base = img_name_list[i_test].split(os.sep)[-1].rsplit(".", 1)[0]
        output_svg = os.path.join(os.getcwd(), "outputs", "svg", nombre_base + ".svg")

        contour = extract_main_contour(mask_for_cv2)
        contour_to_svg(contour, output_svg, canvas_size=(w, h))

        # Limpieza de memoria (AL FINAL del loop)
        del d1, d2, d3, d4, d5, d6, d7

    # OBTENER DETALLES INTERNO DE PERSONAS

    


if __name__ == "__main__":
    main()
