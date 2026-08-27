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
        image_path (str | Path): Path to the input image.
    Returns:
        numpy.ndarray: Image in BGR format.
    Raises:
        FileNotFoundError: If the image cannot be loaded.
    """
    image_bgr = cv2.imread(str(image_path))

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
    Resizes and pads an image to TARGET_SIZE x TARGET_SIZE maintaining aspect ratio.
    Args:
        image_bgr (numpy.ndarray):
            Input image in BGR format.
    Returns:
        numpy.ndarray:
            Resized image with dimensions TARGET_SIZE x TARGET_SIZE.
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

def segment_person_interactive(image_bgr, predictor):
    """
    Interactive segmentation using OpenCV GUI window.
    Controls:
        - Left mouse button: Add a new segmentation region.
        - Z: Undo the last selected region.
        - S: Save the final mask and continue.
        - Esc: Cancel the segmentation.
    """
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
    
    cv2.setMouseCallback("Image", click_event)
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

def segment_person_automatic(image_bgr, predictor):
    """
    Automatic headless segmentation using SAM prompts placed at salient body regions.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    
    h, w = image_bgr.shape[:2]
    
    # Guide points at center and vertical thirds
    input_points = np.array([
        [w // 2, h // 2],
        [w // 2, h // 3],
        [w // 2, 2 * (h // 3)]
    ])
    input_labels = np.array([1, 1, 1])

    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True
    )

    # Select the mask with highest confidence score
    best_mask = masks[np.argmax(scores)].astype(np.uint8) * 255
    
    # Morphological cleaning
    kernel = np.ones((7, 7), np.uint8)
    best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE, kernel)
    best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_OPEN, kernel)
    
    return best_mask

# MAIN PREPROCESSING FUNCTION
def preprocess(image_input, predictor=None, interactive=True):
    """
    Preprocesses an input image before it is fed into the U-Net.
    Supports both file paths and in-memory numpy arrays, in interactive or automatic mode.

    Args:
        image_input (str | Path | np.ndarray):
            Path to image file or decoded BGR image array.
        predictor (SamPredictor, optional):
            Preloaded SAM predictor. If None, it will be loaded automatically.
        interactive (bool):
            If True, uses interactive OpenCV window. If False, runs automatic segmentation.

    Returns:
        tuple (torch.Tensor, np.ndarray):
            Preprocessed tensor ready for U-Net [3, 512, 512] and resized BGR image.
    """
    if predictor is None:
        predictor = load_sam()

    # Handle image input (path vs array)
    if isinstance(image_input, (str, Path)):
        image_bgr = load_image(image_input)
    elif isinstance(image_input, np.ndarray):
        image_bgr = image_input
    else:
        raise TypeError(f"Unsupported image_input type: {type(image_input)}")

    # Segment based on mode
    if interactive:
        mask = segment_person_interactive(image_bgr, predictor)
    else:
        mask = segment_person_automatic(image_bgr, predictor)

    segmented = apply_mask(image_bgr, mask)
    resized = resize_image(segmented)
    image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    tensor = input_transform(image_pil)
    return tensor, resized