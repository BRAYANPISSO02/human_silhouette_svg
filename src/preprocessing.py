import  os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from segment_anything import sam_model_registry, SamPredictor

#Request a ticket
# Ocultar ventana principal de tkinter
root = tk.Tk()
root.withdraw()

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

filename = "foto.png"
output_path = os.path.join(OUTPUT_DIR, filename)
cv2.imwrite(output_path, image) 


# Paths definition
CHECKPOINT = "segment_anything\sam_vit_h_4b8939.pth"
INPUT_IMAGE = "data/input_segmentation/foto.png"
OUTPUT_IMAGE = "data/output_segmentation/foto.png"
OUTPUT = "data/input_train/"
OUTPUT_OUTPUT_IMAGE = "data/output_train/"
os.makedirs(OUTPUT, exist_ok=True)

save_index = 1

#We imported the image
image_bgr = cv2.imread(INPUT_IMAGE)
output_image_bgr = cv2.imread(OUTPUT_IMAGE)

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
def number_to_name(n):
    names = [
        "primera", "segunda", "tercera", "cuarta", "quinta",
        "sexta", "séptima", "octava", "novena", "décima"
    ]
    return names[n - 1] if n <= len(names) else f"imagen_{n}"

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

    # Pressing go out
    elif key == 27:
        break

cv2.destroyAllWindows()