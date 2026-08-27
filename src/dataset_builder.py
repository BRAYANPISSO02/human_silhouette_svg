import sys
import os
from pathlib import Path
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog
from segment_anything import sam_model_registry, SamPredictor
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .config import (
        STATE_FILE,
        INPUT_SEG_DIR,
        OUTPUT_SEG_DIR,
        TRAIN_INPUT_DIR,
        TRAIN_OUTPUT_DIR,
        SAM_CHECKPOINT
    )
except ImportError:
    from src.config import (
        STATE_FILE,
        INPUT_SEG_DIR,
        OUTPUT_SEG_DIR,
        TRAIN_INPUT_DIR,
        TRAIN_OUTPUT_DIR,
        SAM_CHECKPOINT
    )

def load_last_index():
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f).get("last_index", 0)
    return 0

def save_last_index(index):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump({"last_index": index}, f)

def ask_start_index():
    last = load_last_index()
    next_auto = last + 1

    dialog = tk.Tk()
    dialog.title("Iniciar preprocessing")
    dialog.resizable(False, False)
    dialog.geometry("320x160")

    tk.Label(dialog, text=f"Último índice guardado: {last}", font=("Arial", 10)).pack(pady=(16, 4))
    tk.Label(dialog, text="Iniciar desde (dejar vacío = continuar):", font=("Arial", 10)).pack()

    entry = tk.Entry(dialog, font=("Arial", 12), justify="center", width=10)
    entry.pack(pady=6)

    result = [next_auto]  # valor por defecto

    def on_continue():
        result[0] = next_auto
        dialog.destroy()

    def on_custom():
        val = entry.get().strip()
        if val.isdigit() and int(val) > 0:
            result[0] = int(val)
        else:
            result[0] = next_auto
        dialog.destroy()

    btn_frame = tk.Frame(dialog)
    btn_frame.pack(pady=8)
    tk.Button(btn_frame, text=f"Continuar desde {next_auto}", command=on_continue, width=18).pack(side="left", padx=6)
    tk.Button(btn_frame, text="Usar número ingresado", command=on_custom, width=18).pack(side="left", padx=6)

    dialog.mainloop()
    return result[0]

START_FROM = ask_start_index()
print(f"Arrancando desde foto{START_FROM}.png")

# Abrir explorador de archivos
INPUT_IMAGE = filedialog.askopenfilename(
    title="Selecciona una imagen",
    filetypes=[("Imágenes", "*.png *.jpg *.jpeg")]
)
if not INPUT_IMAGE:
    print("No se seleccionó ninguna imagen.")
    exit()

# Cargar imagen
image = cv2.imread(INPUT_IMAGE)
if image is None:
    print("Error al leer la imagen.")
    exit()
print("Imagen cargada:", INPUT_IMAGE)

# OUTPUT INPUT SEGMENTATION
INPUT_SEG_DIR.mkdir(parents=True, exist_ok=True)
filename = f"foto{START_FROM}.png"
output_path = INPUT_SEG_DIR / filename
cv2.imwrite(str(output_path), image)

def load_image_pair(index):
    input_path = INPUT_SEG_DIR / f"foto{index}.png"
    output_path = OUTPUT_SEG_DIR / f"foto{index}.png"
    
    img_input = cv2.imread(str(input_path))
    img_output = cv2.imread(str(output_path))
    
    if img_input is None:
        print(f"No se encontró imagen de entrada: {input_path}")
        exit()
    if img_output is None:
        print(f"No se encontró imagen de salida: {output_path}")
        exit()
        
    print(f"Cargado par: foto{index}.png")
    return img_input, img_output

image_bgr, output_image_bgr = load_image_pair(START_FROM)

TRAIN_INPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Function to get the next save index based on existing files in the output directory
def get_next_save_index(output_dir):
    i = 1
    while (output_dir / f"imagen_{i}.png").exists():
        i += 1
    return i

save_index = get_next_save_index(TRAIN_INPUT_DIR)

# Adjust image to neural network input
def resize_image(image_bgr):
	TARGET_SIZE = 512
	h, w = image_bgr.shape[:2]
	scale = TARGET_SIZE / max(h, w)
	new_w = int(w * scale)
	new_h = int(h * scale)
	resized = cv2.resize(image_bgr, (new_w, new_h))
	image_bgr = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
	x_offset = (TARGET_SIZE - new_w) // 2
	y_offset = (TARGET_SIZE - new_h) // 2
	image_bgr[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
	return image_bgr

output_image_bgr = resize_image(output_image_bgr)
image_bgr = resize_image(image_bgr)

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# We load the model
if not SAM_CHECKPOINT.exists():
    raise FileNotFoundError(f"No se encontró el checkpoint de SAM en: {SAM_CHECKPOINT}")

sam = sam_model_registry["vit_h"](checkpoint=str(SAM_CHECKPOINT))
predictor = SamPredictor(sam)
predictor.set_image(image_rgb)

# Storage
masks_stack = []  # key to undo
mask_total = np.zeros(image_bgr.shape[:2], dtype=np.uint8)

# convert number to name
def number_to_name(n):
    return f"imagen_{n}"

# reconstruction
def rebuild_mask():
    global mask_total
    mask_total = np.zeros_like(mask_total)

    for m in masks_stack: # It overlays all the masks stored in a bitwise "or" comparator 
        # to leave a final one with all the active 1 parts.
        mask_total = cv2.bitwise_or(mask_total, m)
    # global cleanup
    kernel = np.ones((7,7), np.uint8)
    mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kernel)
    mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel)
    
# Click event
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:

        input_point = np.array([[x, y]])
        input_label = np.array([1])

		# We used the SAM predictor
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )

        mask = masks[np.argmax(scores)].astype(np.uint8) * 255

        # Individual cleaning
        kernel = np.ones((5,5), np.uint8)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)

        # Save to stack
        masks_stack.append(mask_clean)

        rebuild_mask()

        cv2.imshow("Mask total", mask_total)
        print("Mask added")

# Printing and function activation
cv2.imshow("Image", image_bgr)
cv2.setMouseCallback("Image", click_event)

print("Controls:")
print("Left click: add mask")
print("Z: Undo last")
print("S: Save result")
print("ESC: Go out")

# Current reading
while True:
    key = cv2.waitKey(1) & 0xFF

    # Pressing undo
    if key == ord('z'):
        if masks_stack:
            masks_stack.pop()
            rebuild_mask()
            cv2.imshow("Mask Total", mask_total)
            print("Last mask removed")

    # Pressing save
    elif key == ord('s'):

        name = number_to_name(save_index)
        input_path = TRAIN_INPUT_DIR / f"{name}mask.png"
        input_mask_path = TRAIN_INPUT_DIR / f"{name}.png"
        output_path = TRAIN_OUTPUT_DIR / f"{name}.png"
        cv2.imwrite(str(input_path), mask_total)
        segmented = cv2.bitwise_and(image_bgr, image_bgr, mask=mask_total)
        cv2.imwrite(str(input_mask_path), segmented)
        cv2.imwrite(str(output_path), output_image_bgr)
        print(f"Guardado: {name}")

        save_index += 1   
        save_last_index(START_FROM)
    
    elif key == 27:
        save_last_index(START_FROM)  # ← También guarda al salir
        break

cv2.destroyAllWindows()