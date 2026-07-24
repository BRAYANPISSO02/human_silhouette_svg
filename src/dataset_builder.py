import  os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog
from segment_anything import sam_model_registry, SamPredictor
import json 

STATE_FILE = "data/preprocessing_state.json"

def load_last_index():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f).get("last_index", 0)
    return 0

def save_last_index(index):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
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

# OUTPUT
OUTPUT_DIR = "data/input_segmentation/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

filename = f"foto{START_FROM}.png"
output_path = os.path.join(OUTPUT_DIR, filename)
cv2.imwrite(output_path, image)


# Paths definition
INPUT_IMAGE_DIR = "data/input_segmentation/"
OUTPUT_IMAGE_DIR = "data/output_segmentation/"

def load_image_pair(index):
    input_path = os.path.join(INPUT_IMAGE_DIR, f"foto{index}.png")
    output_path = os.path.join(OUTPUT_IMAGE_DIR, f"foto{index}.png")
    
    img_input = cv2.imread(input_path)
    img_output = cv2.imread(output_path)
    
    if img_input is None:
        print(f"No se encontró imagen de entrada: {input_path}")
        exit()
    if img_output is None:
        print(f"No se encontró imagen de salida: {output_path}")
        exit()
        
    print(f"Cargado par: foto{index}.png")
    return img_input, img_output

image_bgr, output_image_bgr = load_image_pair(START_FROM)

CHECKPOINT = "segment_anything\sam_vit_h_4b8939.pth"
OUTPUT = "data/input_train/"
OUTPUT_OUTPUT_IMAGE = "data/output_train/"
os.makedirs(OUTPUT, exist_ok=True)

# Function to get the next save index based on existing files in the output directory
def get_next_save_index(output_dir):
    i = 1
    while os.path.exists(os.path.join(output_dir, f"imagen_{i}.png")):
        i += 1
    return i

save_index = get_next_save_index(OUTPUT)

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
sam = sam_model_registry["vit_h"](checkpoint=CHECKPOINT)
predictor = SamPredictor(sam)
predictor.set_image(image_rgb)

# Storage
masks_stack = []  # key to undo
mask_total = np.zeros(image_bgr.shape[:2], dtype=np.uint8)

# convert number to name
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
        input_path = os.path.join(OUTPUT, f"{name}mask.png")
        input_mask_path = os.path.join(OUTPUT, f"{name}.png")
        output_path = os.path.join(OUTPUT_OUTPUT_IMAGE, f"{name}.png")
        cv2.imwrite(input_path, mask_total)
        segmented = cv2.bitwise_and(image_bgr, image_bgr, mask=mask_total)
        cv2.imwrite(input_mask_path, segmented)
        cv2.imwrite(output_path, output_image_bgr)
        print(f"Guardado: {name}")

        save_index += 1   
        save_last_index(START_FROM)
    
    elif key == 27:
        save_last_index(START_FROM)  # ← También guarda al salir
        break  

    # Pressing go out
    elif key == 27:
        break

cv2.destroyAllWindows()