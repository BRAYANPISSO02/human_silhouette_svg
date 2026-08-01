import torch

from PIL import Image
from .utils import input_transform
from .preprocessing import load_image, load_sam
from .training import UNet
from .postprocessing import postprocess
from .vectorization import vectorize

CHECKPOINT = (r"C:\Users\Laboratorio\Desktop\proyecto_siluetas\human_silhouette_svg\checkpoints\best_model.pth")

predictor = load_sam()
image = load_image(r"C:\Users\Laboratorio\Desktop\proyecto_siluetas\human_silhouette_svg\data\input_val\imagen_77.png")

print(image.shape)

# --------------------------------------------------
# Device used for inference
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------------------------------
# Create the model
# --------------------------------------------------
model = UNet().to(device)
model.eval()
print("U-Net creada correctamente.")

# --------------------------------------------------
# Prepare your U-Net entrance exam application.
# --------------------------------------------------
image_pil = Image.fromarray(image)
input_tensor = input_transform(image_pil)
input_tensor = input_tensor.unsqueeze(0)   # [1,3,512,512]
input_tensor = input_tensor.to(device)
print(input_tensor.shape)

# --------------------------------------------------
# Lift trained weights
# --------------------------------------------------
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()
print("Pesos del modelo cargados correctamente.")

# --------------------------------------------------
# Inference
# --------------------------------------------------
with torch.no_grad():
    prediction = model(input_tensor)
print(prediction.shape)

# --------------------------------------------------
# Postprocessing
# --------------------------------------------------
binary_image = postprocess(prediction)

# --------------------------------------------------
# Vectorization
# --------------------------------------------------
svg_path = vectorize(binary_image, "image_77.svg")
print(f"SVG generado: {svg_path}")