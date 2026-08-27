import sys
import os
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
    from .preprocessing import preprocess
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
    from src.preprocessing import preprocess
    from src.training import UNet
    from src.postprocessing import postprocess
    from src.vectorization import vectorize

# Open browser to select image
def select_image():
    """
    Opens a file selection dialog and allows the user to choose an input image.
    The selected image path is returned and later used as the input of the
    preprocessing pipeline.
    Returns:
        str:
            Absolute path of the selected image file.
    Raises:
        ValueError:
            If no image is selected and the dialog is closed or cancelled.
    """
    root = Tk()
    root.withdraw()  # Oculta la ventana principal
    root.attributes("-topmost", True)
    image_path = askopenfilename(
        title="Seleccionar imagen",
        filetypes=[
            ("Imágenes", "*.png *.jpg *.jpeg *.bmp"),
            ("Todos los archivos", "*.*")])
    root.destroy()
    if not image_path:
        raise ValueError("No se seleccionó ninguna imagen.")
    return image_path

def main():
    """
    Executes the complete human silhouette generation pipeline.

    The pipeline includes image selection, preprocessing, U-Net inference,
    postprocessing, vectorization, and saving the outputs generated at each
    stage.
    """

    # Create output directories
    ensure_output_dirs()

    # Device used for inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the model
    model = UNet().to(device)
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el checkpoint del modelo en: {BEST_MODEL_PATH}")

    model.load_state_dict(torch.load(str(BEST_MODEL_PATH), map_location=device))
    model.eval()
    print("Pesos del modelo cargados correctamente.")

    # Load and preprocess the input image
    image_path = select_image()
    # Base name used for all output files
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    input_tensor, preprocessed_image = preprocess(image_path)
    input_tensor = input_tensor.unsqueeze(0).to(device)  # [1,3,512,512]

    # Save the preprocessed image
    preprocessing_path = OUTPUT_PREPROCESSING_DIR / f"{base_name}.png"
    cv2.imwrite(str(preprocessing_path), preprocessed_image)
    print(f"Preprocessing guardada en: {preprocessing_path}")

    # Inference
    with torch.no_grad():
        prediction = model(input_tensor)
    print(f"Inference output shape: {prediction.shape}")

    # Save U-Net output
    unet_output = prediction.squeeze().detach().cpu().numpy()
    # Convert from [-1,1] to [0,255]
    unet_output = ((unet_output + 1.0) / 2.0) * 255.0
    unet_output = unet_output.clip(0, 255).astype("uint8")
    unet_path = OUTPUT_UNET_DIR / f"{base_name}.png"
    Image.fromarray(unet_output).save(str(unet_path))
    print(f"U-Net guardada en: {unet_path}")

    # Postprocessing
    binary_image = postprocess(prediction)

    # Save postprocessing result
    postprocessing_path = OUTPUT_POSTPROCESSING_DIR / f"{base_name}.png"
    Image.fromarray(binary_image).save(str(postprocessing_path))
    print(f"Postprocessing guardado en: {postprocessing_path}")

    # Vectorization and save result
    vectorization_path = OUTPUT_VECTORIZATION_DIR / f"{base_name}.svg"
    svg_path = vectorize(binary_image, str(vectorization_path))
    print(f"SVG guardado en: {svg_path}")

if __name__ == "__main__":
    main()