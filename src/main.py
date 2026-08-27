import sys
import os
import uuid
from pathlib import Path
import torch
import cv2
from PIL import Image
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Add project root to sys.path so it can be run directly or as a module
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .config import (
        BEST_MODEL_PATH,
        OUTPUT_PREPROCESSING_DIR,
        OUTPUT_UNET_DIR,
        OUTPUT_POSTPROCESSING_DIR,
        OUTPUT_VECTORIZATION_DIR,
        ensure_output_dirs
    )
    from .preprocessing import preprocess, load_sam
    from .training import UNet
    from .postprocessing import postprocess
    from .vectorization import vectorize
except ImportError:
    from src.config import (
        BEST_MODEL_PATH,
        OUTPUT_PREPROCESSING_DIR,
        OUTPUT_UNET_DIR,
        OUTPUT_POSTPROCESSING_DIR,
        OUTPUT_VECTORIZATION_DIR,
        ensure_output_dirs
    )
    from src.preprocessing import preprocess, load_sam
    from src.training import UNet
    from src.postprocessing import postprocess
    from src.vectorization import vectorize

def load_unet_model(device=None):
    """
    Loads the trained U-Net model from the checkpoint file.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {BEST_MODEL_PATH}")

    model = UNet().to(device)
    model.load_state_dict(torch.load(str(BEST_MODEL_PATH), map_location=device))
    model.eval()
    return model

def run_pipeline(
    image_input,
    model=None,
    predictor=None,
    interactive=True,
    save_intermediates=False,
    base_name=None,
    device=None
):
    """
    Executes the full photo-to-SVG vectorization pipeline.
    Reused by both the desktop CLI (interactive) and the web API (automatic).

    Args:
        image_input (str | Path | np.ndarray): Image path or decoded BGR array.
        model (UNet, optional): Preloaded U-Net model.
        predictor (SamPredictor, optional): Preloaded SAM predictor.
        interactive (bool): Whether to show interactive OpenCV GUI.
        save_intermediates (bool): Whether to save debug images to outputs/.
        base_name (str, optional): Filename prefix for output files.
        device (torch.device, optional): CPU or CUDA device.

    Returns:
        tuple (str, str): Path to generated SVG file and SVG file content string.
    """
    ensure_output_dirs()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model = load_unet_model(device)

    if base_name is None:
        if isinstance(image_input, (str, Path)):
            base_name = Path(image_input).stem
        else:
            base_name = f"silhouette_{uuid.uuid4().hex[:8]}"

    # Step 1: Preprocessing (SAM segmentation + letterboxing)
    input_tensor, preprocessed_image = preprocess(
        image_input,
        predictor=predictor,
        interactive=interactive
    )
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # Save intermediate preprocessing image if requested
    if save_intermediates:
        preprocessing_path = OUTPUT_PREPROCESSING_DIR / f"{base_name}.png"
        cv2.imwrite(str(preprocessing_path), preprocessed_image)
        print(f"Preprocessing saved to: {preprocessing_path}")

    # Step 2: U-Net Inference
    with torch.no_grad():
        prediction = model(input_tensor)

    # Save intermediate U-Net output if requested
    if save_intermediates:
        unet_output = prediction.squeeze().detach().cpu().numpy()
        unet_output = ((unet_output + 1.0) / 2.0) * 255.0
        unet_output = unet_output.clip(0, 255).astype("uint8")
        unet_path = OUTPUT_UNET_DIR / f"{base_name}.png"
        Image.fromarray(unet_output).save(str(unet_path))
        print(f"U-Net output saved to: {unet_path}")

    # Step 3: Postprocessing and Skeletonization
    binary_image = postprocess(prediction)

    if save_intermediates:
        postprocessing_path = OUTPUT_POSTPROCESSING_DIR / f"{base_name}.png"
        Image.fromarray(binary_image).save(str(postprocessing_path))
        print(f"Postprocessing saved to: {postprocessing_path}")

    # Step 4: Potrace Vectorization to SVG
    vectorization_path = OUTPUT_VECTORIZATION_DIR / f"{base_name}.svg"
    svg_path = vectorize(binary_image, str(vectorization_path))
    print(f"SVG saved to: {svg_path}")

    # Read SVG content
    with open(svg_path, "r", encoding="utf-8") as f:
        svg_content = f.read()

    return str(svg_path), svg_content

def select_image():
    """
    Opens a file selection dialog to choose an input image for desktop mode.
    """
    root = Tk()
    root.withdraw()  # Hide main window
    root.attributes("-topmost", True)
    image_path = askopenfilename(
        title="Seleccionar imagen",
        filetypes=[
            ("Imágenes", "*.png *.jpg *.jpeg *.bmp"),
            ("Todos los archivos", "*.*")
        ]
    )
    root.destroy()
    if not image_path:
        raise ValueError("No se seleccionó ninguna imagen.")
    return image_path

def main():
    """
    Desktop entry point: Selects an image with Tkinter and executes the pipeline with GUI.
    """
    image_path = select_image()
    run_pipeline(image_path, interactive=True, save_intermediates=True)

if __name__ == "__main__":
    main()