import os
import cv2
import numpy as np
from segmentation import segment_foreground_grabcut
from preprocessing import preprocess_for_segmentation
from preprocessing import preprocess_for_ml 
from postprocessing import clean_mask
from vectorization import extract_main_contour, contour_to_svg

def load_image(path: str) -> np.ndarray:
    """
    Load an image from disk.

    Args:
        path (str): Path to input image.

    Returns:
        np.ndarray: Loaded image.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(path)
    if image is None:
        raise RuntimeError(f"Failed to load image: {path}")

    return image

def image_to_silhouette_svg(image_path: str, output_svg: str) -> None:
    """
    Full pipeline: image -> silhouette SVG.
    """
    image = load_image(image_path)
    image_pre = preprocess_for_segmentation(image, size=512)
    
    os.makedirs("outputs/debug", exist_ok=True)
    cv2.imwrite("outputs/debug/preprocessed.png", image_pre)
    
    mask = segment_foreground_grabcut(image)
    mask = clean_mask(mask)

    contour = extract_main_contour(mask)

    h, w = mask.shape
    contour_to_svg(contour, output_svg, canvas_size=(w, h))

if __name__ == "__main__":
    input_image = "data/photo/persona_2.jpg"
    output_svg = "outputs/svg/silhouette.svg"
    os.makedirs("outputs/svg", exist_ok=True)

    image_to_silhouette_svg(input_image, output_svg)

    print("[INFO] Silhouette SVG generated successfully.")