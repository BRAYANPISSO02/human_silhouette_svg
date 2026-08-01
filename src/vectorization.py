import os
import cv2
import numpy as np

def vectorize(binary_image: np.ndarray, output_name: str = "result.svg"):

    contours, _ = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    height, width = binary_image.shape

    svg = []

    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" '
        f'height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )

    svg.append('<g fill="none" stroke="black" stroke-width="1">')

    for contour in contours:

        if cv2.contourArea(contour) < 10:
            continue

        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) < 2:
            continue

        path = []

        first = approx[0][0]
        path.append(f"M {first[0]} {first[1]}")

        for point in approx[1:]:
            x, y = point[0]
            path.append(f"L {x} {y}")

        path.append("Z")

        svg.append(f'<path d="{" ".join(path)}"/>')

    svg.append("</g>")
    svg.append("</svg>")

    # --------------------------------------------------
    # Guardar SVG en outputs/svg
    # --------------------------------------------------

    output_dir = os.path.join("outputs", "svg")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, output_name)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))

    return output_path