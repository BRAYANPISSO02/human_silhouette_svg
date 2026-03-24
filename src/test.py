import cv2
import numpy as np

input_png = r"F:\proyectos_independientes\human_silhouette_svg\data\photo\140_large-img_1616_fake_B.png"
output_png = r"F:\proyectos_independientes\human_silhouette_svg\outputs\debug\prueba.png"

def skeletonize_image(image_path, output_path=None, debug=False):
    # 1. Leer imagen
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("No se pudo cargar la imagen")

    # 2. Escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Suavizado
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4. Detección de bordes (Canny)
    edges = cv2.Canny(blur, 30, 100)

    # 5. Dilatar para conectar líneas
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # 6. Esqueletización
    skeleton = np.zeros(edges.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    img_bin = edges.copy()

    while True:
        eroded = cv2.erode(img_bin, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img_bin, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img_bin = eroded.copy()

        if cv2.countNonZero(img_bin) == 0:
            break

    # 7. Invertir (líneas negras, fondo blanco)
    skeleton = cv2.bitwise_not(skeleton)

    # 8. Guardar resultado
    if output_path:
        cv2.imwrite(output_path, skeleton)

    return skeleton


# Ejecutar
skeleton = skeletonize_image(input_png, output_png, debug=True)