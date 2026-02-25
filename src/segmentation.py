import numpy as np
import cv2
from ultralytics import YOLO

# from skimage import color, filters, morphology


# def segment_person(
#     image_rgb,
#     min_object_size=800,
#     min_hole_size=800,
#     closing_radius=3
# ):
#     """
#     Segmenta la persona desde una imagen RGB ya preprocesada y centrada.

#     Parámetros
#     ----------
#     image_rgb : np.ndarray
#         Imagen RGB (H, W, 3)
#     min_object_size : int
#         Área mínima para conservar regiones (ruido eliminado)
#     min_hole_size : int
#         Área mínima de huecos a rellenar dentro del cuerpo
#     closing_radius : int
#         Radio del elemento estructurante para cierre morfológico

#     Retorna
#     -------
#     mask : np.ndarray (bool)
#         Máscara binaria (True = persona, False = fondo)
#     """

#     # 1. Conversión a escala de grises
#     gray = color.rgb2gray(image_rgb)

#     # 2. Umbral automático (Otsu)
#     threshold = filters.threshold_otsu(gray)
#     mask = gray < threshold  # persona en blanco

#     # 3. Eliminar regiones pequeñas (ruido)
#     mask = morphology.remove_small_objects(
#         mask,
#         min_size=min_object_size
#     )

#     # 4. Rellenar huecos dentro del cuerpo
#     mask = morphology.remove_small_holes(
#         mask,
#         area_threshold=min_hole_size
#     )

#     # 5. Cierre morfológico para solidez
#     selem = morphology.disk(closing_radius)
#     mask = morphology.binary_closing(mask, selem)

#     return mask


class PersonSegmenterYOLO:
    def __init__(self, model_path: str = "yolov8n-seg.pt", conf_threshold: float = 0.4):
        """
        YOLOv8 person segmenter.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def segment_all_people(self, image: np.ndarray) -> np.ndarray:
        """
        Returns a single binary mask containing ALL detected people.
        White = person, Black = background
        """
        results = self.model(image, verbose=False)[0]

        if results.masks is None:
            return np.zeros(image.shape[:2], dtype=np.uint8)

        final_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for mask, cls, conf in zip(
            results.masks.data, results.boxes.cls, results.boxes.conf
        ):
            if int(cls) != 0:  # class 0 = person
                continue

            if conf < self.conf_threshold:
                continue

            mask_np = mask.cpu().numpy()
            mask_bin = (mask_np > 0.5).astype(np.uint8) * 255

            mask_bin = cv2.resize(
                mask_bin,
                (final_mask.shape[1], final_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

            final_mask = np.maximum(final_mask, mask_bin)

        return final_mask
