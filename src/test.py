import os
import cv2
import torch
from PIL import Image

from .training import UNet
from .utils import input_transform, denormalize


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

CHECKPOINT = r"C:\Users\Laboratorio\Desktop\proyecto_siluetas\human_silhouette_svg\checkpoints\best_model.pth"

INPUT_IMAGE = r"C:\Users\Laboratorio\Desktop\proyecto_siluetas\human_silhouette_svg\data\input_train\imagen_1.png"

OUTPUT_DIR = r"C:\Users\Laboratorio\Desktop\proyecto_siluetas\human_silhouette_svg\data\test"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = UNet().to(device)

model.load_state_dict(
    torch.load(
        CHECKPOINT,
        map_location=device
    )
)

model.eval()

print("Model loaded successfully.")


# --------------------------------------------------
# LOAD IMAGE
# --------------------------------------------------

image = Image.open(INPUT_IMAGE)

input_tensor = input_transform(image)

input_tensor = input_tensor.unsqueeze(0).to(device)


# --------------------------------------------------
# INFERENCE
# --------------------------------------------------

with torch.no_grad():

    prediction = model(input_tensor)


# --------------------------------------------------
# CONVERT OUTPUT
# --------------------------------------------------

prediction = denormalize(prediction)

prediction_path = os.path.join(
    OUTPUT_DIR,
    "prediction.png"
)

cv2.imwrite(
    prediction_path,
    prediction
)

print(f"Prediction saved: {prediction_path}")


# --------------------------------------------------
# SHOW RESULT
# --------------------------------------------------

original = cv2.imread(INPUT_IMAGE)

result = cv2.imread(
    prediction_path,
    cv2.IMREAD_GRAYSCALE
)

cv2.imshow("Input", original)
cv2.imshow("Prediction", result)

cv2.waitKey(0)
cv2.destroyAllWindows()