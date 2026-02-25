import os
import cv2
import numpy as np
from segmentation import PersonSegmenterYOLO
from preprocessing import preprocess_for_segmentation, preprocess_for_ml

# from hog_person_detector import detect_and_reframe_person
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
    image = preprocess_for_segmentation(image, size=1024)

    os.makedirs("outputs/debug", exist_ok=True)
    cv2.imwrite("outputs/debug/preprocessed.png", image)

    # image = detect_and_reframe_person(image)

    # cv2.imwrite("outputs/debug/center.png", image)

    segmenter = PersonSegmenterYOLO(model_path="yolov8n-seg.pt", conf_threshold=0.4)

    mask = segmenter.segment_all_people(image)
    cv2.imwrite("outputs/debug/mask.png", mask)

    # mask = segment_person(image)

    # mask = (mask * 255).astype(np.uint8)
    # cv2.imwrite("outputs/debug/mask.png", mask)

    mask = clean_mask(mask)
    cv2.imwrite("outputs/debug/mask_clean.png", mask)

    # contour = extract_main_contour(mask)

    # h, w = mask.shape
    # contour_to_svg(contour, output_svg, canvas_size=(w, h))


if __name__ == "__main__":
    input_image = "data/photo/persona_3.jpg"
    output_svg = "outputs/svg/silhouette.svg"
    os.makedirs("outputs/svg", exist_ok=True)

    image_to_silhouette_svg(input_image, output_svg)

    print("[INFO] Silhouette SVG generated successfully.")
