import numpy as np
import torch
from PIL import Image
from torchvision import transforms


# ─────────────────────────────────────────────
# INPUT TRANSFORM  (foto RGB → tensor)
# ─────────────────────────────────────────────
# Receives: PNG 512x512, 3 channels (BGR already converted to RGB in preprocessing)
# Returns:  tensor [3, 512, 512] with values between -1 and 1

input_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),  # handles alpha channel if present
    transforms.ToTensor(),                           # [0,255] uint8 → [0.0, 1.0] float32
    transforms.Normalize(                            # [0.0, 1.0] → [-1.0, 1.0]
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# ─────────────────────────────────────────────
# OUTPUT TRANSFORM  (line art → tensor)
# ─────────────────────────────────────────────
# Receives: PNG 512x512, black background white lines
# Returns:  tensor [1, 512, 512] with values between -1 and 1

output_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),    # ensures 1 channel (black & white)
    transforms.ToTensor(),                           # [0,255] uint8 → [0.0, 1.0] float32
    transforms.Normalize(                            # [0.0, 1.0] → [-1.0, 1.0]
        mean=[0.5],
        std=[0.5]
    )
])


# ─────────────────────────────────────────────
# DENORMALIZE  (tensor → image)
# ─────────────────────────────────────────────
# Receives: tensor [1, 512, 512] with values between -1 and 1 (model output)
# Returns:  numpy array [512, 512] with values between 0 and 255
# Used in: main.py to convert model prediction into a visible/saveable image

def denormalize(tensor):
    """
    Reverts normalization applied during training.
    Converts model output tensor back to a PNG-compatible image.

    Args:
        tensor: torch.Tensor [1, H, W] or [1, 1, H, W], values in [-1, 1]

    Returns:
        numpy array [H, W], dtype uint8, values in [0, 255]
    """
    # Remove batch dimension if present [1, 1, H, W] → [1, H, W]
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    # Revert normalization: [-1, 1] → [0.0, 1.0]
    tensor = tensor * 0.5 + 0.5

    # Clamp to valid range to avoid artifacts from floating point errors
    tensor = torch.clamp(tensor, 0.0, 1.0)

    # Tensor [1, H, W] → numpy [H, W] → scale to [0, 255]
    image = tensor.squeeze(0).cpu().numpy()
    image = (image * 255).astype(np.uint8)

    return image