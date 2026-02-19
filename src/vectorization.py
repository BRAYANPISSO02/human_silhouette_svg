import cv2
import numpy as np
import svgwrite

def extract_main_contour(mask: np.ndarray) -> np.ndarray:
    """
    Extract the largest contour from a binary mask.
    """
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise RuntimeError("No contours found")

    return max(contours, key=cv2.contourArea)


def contour_to_svg(
    contour: np.ndarray,
    svg_path: str,
    canvas_size: tuple[int, int]
) -> None:
    """
    Convert a contour into an SVG path.
    """
    dwg = svgwrite.Drawing(svg_path, size=canvas_size)

    path_cmds = []
    start = contour[0][0]
    path_cmds.append(f"M {start[0]} {start[1]}")

    for point in contour[1:]:
        x, y = point[0]
        path_cmds.append(f"L {x} {y}")

    path_cmds.append("Z")

    dwg.add(
        dwg.path(
            d=" ".join(path_cmds),
            fill="black",
            stroke="none"
        )
    )

    dwg.save()