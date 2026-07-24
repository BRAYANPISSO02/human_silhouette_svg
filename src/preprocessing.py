#IMPORTS
import cv2
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor

from.utils import input_transform

#CONSTANTS
CHECKPOINT = "segment_anything/sam_vit_h_4b8939.pth"
TARGET_SIZE = 512

#SECONDARY FUNCTIONS
def load_image(image_path):
    """
    Loads an image from disk.
    Args:
        image_path (str): Path to the input image.
    Returns:
        numpy.ndarray: Image in BGR format.
    Raises:
        FileNotFoundError: If the image cannot be loaded.
    """
    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )
    return image_bgr

def load_sam():
    """
    Loads the Segment Anything Model (SAM).
    Returns:
        SamPredictor: Initialized SAM predictor.
    """
    sam = sam_model_registry["vit_h"](checkpoint=CHECKPOINT)
    predictor = SamPredictor(sam)
    return predictor

def segment_person(image_bgr, predictor):

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    masks_stack = []
    mask_total = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    def rebuild_mask():
        nonlocal mask_total
        mask_total = np.zeros_like(mask_total)
        for mask in masks_stack:
            mask_total = cv2.bitwise_or(mask_total, mask)
        kernel = np.ones((7, 7), np.uint8)
        mask_total = cv2.morphologyEx(
            mask_total, 
            cv2.MORPH_CLOSE, 
            kernel
        )
        mask_total = cv2.morphologyEx(
            mask_total, 
            cv2.MORPH_OPEN, 
            kernel
        )

    def click_event(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        input_point = np.array([[x, y]])
        input_label = np.array([1])
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )

        mask = masks[np.argmax(scores)].astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )
        masks_stack.append(mask)
        rebuild_mask()
        cv2.imshow("Mask", mask_total)

    cv2.imshow("Image", image_bgr)
    cv2.imshow("Mask", mask_total)
    cv2.setMouseCallback(
        "Image",
        click_event
    )
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("z"):
            if masks_stack:
                masks_stack.pop()
                rebuild_mask()
                cv2.imshow("Mask", mask_total)
        elif key == ord("s"):
            cv2.destroyAllWindows()
            return mask_total
        elif key == 27:
            cv2.destroyAllWindows()
            return None

#MAIN FUNCTION
def preprocess(image_path):

    image_bgr = load_image(image_path)
    mask = segment_person(image_bgr, predictor)
    segmented = apply_mask(image_bgr, mask)
    resized = resize_image(segmented)
    image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    tensor = input_transform(image_pil)
    return tensor