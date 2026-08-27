import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .config import SAM_CHECKPOINT
    from .utils import input_transform
except ImportError:
    from src.config import SAM_CHECKPOINT
    from src.utils import input_transform

# CONSTANTS
TARGET_SIZE = 512

# SECONDARY FUNCTIONS
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
    if not SAM_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Could not find SAM checkpoint at: {SAM_CHECKPOINT}"
        )
    sam = sam_model_registry["vit_h"](checkpoint=str(SAM_CHECKPOINT))
    predictor = SamPredictor(sam)
    return predictor

def apply_mask(image_bgr, mask):
    """
    Applies the segmentation mask to the image.
    Args:
        image_bgr (numpy.ndarray):
            Original image in BGR format.
        mask (numpy.ndarray):
            Binary mask where the person is 255 and the background is 0.
    Returns:
        numpy.ndarray:
            Image with the background removed.
    Raises:
        ValueError: 
            If no mask is provided.
    """
    if mask is None:
        raise ValueError("Segmentation was cancelled.")
    segmented = cv2.bitwise_and(
        image_bgr,
        image_bgr,
        mask=mask
    )
    return segmented

def resize_image(image_bgr):
    """
    Args:
        image_bgr (numpy.ndarray):
            Input image in BGR format.
    Returns:
        numpy.ndarray:
            Resized image with dimensions TARGET_SIZE × TARGET_SIZE.
    """
    height, width = image_bgr.shape[:2]
    scale = min(
        TARGET_SIZE / width,
        TARGET_SIZE / height
    )
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized = cv2.resize(
        image_bgr,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )
    canvas = np.zeros(
        (TARGET_SIZE, TARGET_SIZE, 3),
        dtype=np.uint8
    )
    x_offset = (TARGET_SIZE - new_width) // 2
    y_offset = (TARGET_SIZE - new_height) // 2
    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = resized
    return canvas

def segment_person(image_bgr, predictor):
    """
    Controls:
        - Left mouse button: Add a new segmentation region.
        - Z: Undo the last selected region.
        - S: Save the final mask and continue.
        - Esc: Cancel the segmentation.
    Args:
        image_bgr (numpy.ndarray):
            Input image in BGR format.
        predictor (SamPredictor):
            Initialized SAM predictor used to generate segmentation masks.
    Returns:
        numpy.ndarray:
            Binary mask where the segmented person has value 255 and the
            background has value 0.
        None:
            If the segmentation is cancelled by pressing the Esc key.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    masks_stack = []
    mask_total = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    def rebuild_mask():
        """
        Rebuilds the final segmentation mask by combining all masks stored in the
        stack and applying morphological operations.

        The resulting mask is stored in the nonlocal variable `mask_total`.
        """
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
        """
        Args:
            event (int):
                OpenCV mouse event identifier.
            x (int):
                X-coordinate of the mouse click.
            y (int):
                Y-coordinate of the mouse click.
            flags (int):
                Additional event flags provided by OpenCV.
            param (Any):
                Optional user-defined data passed by OpenCV.
        Returns:
            None.
        """
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

    # --------------------------------------------------
    # Display windows with a reasonable size
    # --------------------------------------------------
    DISPLAY_SIZE = 800
    h, w = image_bgr.shape[:2]
    scale = min(DISPLAY_SIZE / w, DISPLAY_SIZE / h, 1.0)
    display_width = int(w * scale)
    display_height = int(h * scale)
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", display_width, display_height)
    cv2.resizeWindow("Mask", display_width, display_height)
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
    """
    Preprocesses an input image before it is fed into the neural network.

    The preprocessing pipeline consists of loading the image, interactively
    segmenting the person using SAM, removing the background, resizing the
    segmented image to the target resolution, converting it to RGB and PIL
    format, and applying the input transformations required by the model.
    Args:
        image_path (str):
            Path to the input image.
    Returns:
        torch.Tensor:
            Preprocessed image tensor ready to be used as input to the neural
            network.
    """
    predictor = load_sam()
    image_bgr = load_image(image_path)
    mask = segment_person(image_bgr, predictor)
    segmented = apply_mask(image_bgr, mask)
    resized = resize_image(segmented)
    image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    tensor = input_transform(image_pil)
    return tensor, resized