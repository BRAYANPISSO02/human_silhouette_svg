import sys
import os
from pathlib import Path
import cv2
import subprocess
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .config import POTRACE_PATH
except ImportError:
    from src.config import POTRACE_PATH

def find_contours(binary_image: np.ndarray):
    """
    Finds all contours in a binary image.
    Args:
        binary_image (numpy.ndarray):
            Binary image with white strokes over a black background.
    Returns:
        list:
            List of contours detected in the image.
    """
    contours, _ = cv2.findContours(
        binary_image,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE
    )
    return contours

def filter_contours(contours: list, min_area: float = 8.0):
    """
    Removes very small contours that are likely to be noise.
    Args:
        contours (list):
            List of contours detected in the image.
        min_area (float):
            Minimum contour area to keep.
    Returns:
        list:
            Filtered list of contours.
    """
    filtered = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            filtered.append(contour)
    return filtered

def simplify_contours(contours: list, epsilon_factor: float = 0.0015):
    """
    Simplifies contour geometry while preserving the overall shape.
    Args:
        contours (list):
            List of filtered contours.
        epsilon_factor (float):
            Simplification factor relative to each contour perimeter.
    Returns:
        list:
            List of simplified contours.
    """
    simplified = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        epsilon = epsilon_factor * perimeter
        approx = cv2.approxPolyDP(
            contour,
            epsilon,
            True)
        simplified.append(approx)
    return simplified

def contour_to_svg_path(contour):
    """
    Converts a contour into an SVG path string.
    Args:
        contour (numpy.ndarray):
            Simplified contour.
    Returns:
        str:
            SVG path representation of the contour.
    """
    if len(contour) < 2:
        return ""
    commands = []
    first = contour[0][0]
    commands.append(f"M {first[0]} {first[1]}")
    for point in contour[1:]:
        x, y = point[0]
        commands.append(f"L {x} {y}")
    commands.append("Z")
    return " ".join(commands)

def save_svg(paths: list, width: int, height: int, output_path: str):
    """
    Saves a list of SVG paths to an SVG file.
    Args:
        paths (list):
            List of SVG path strings.
        width (int):
            SVG canvas width.
        height (int):
            SVG canvas height.
        output_path (str):
            Output SVG file path.
    """
    svg = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" '
        f'height="{height}" '
        f'viewBox="0 0 {width} {height}">')
    svg.append(
        '<g fill="none" '
        'stroke="black" '
        'stroke-width="1" '
        'stroke-linecap="round" '
        'stroke-linejoin="round">')
    for path in paths:
        if path:
            svg.append(f'<path d="{path}"/>')
    svg.append("</g>")
    svg.append("</svg>")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))

def vectorize(binary_image: np.ndarray, output_path: str):
    """
    Applies Potrace smoothing and then runs the internal contour-based
    vectorization pipeline.
    """
    if not POTRACE_PATH.exists():
        raise FileNotFoundError(f"No se encontró el ejecutable de Potrace en: {POTRACE_PATH}")

    base_name = os.path.splitext(output_path)[0]
    temp_bmp = base_name + "_potrace_input.bmp"
    temp_pgm = base_name + "_potrace.pgm"
    potrace_input = 255 - binary_image
    cv2.imwrite(temp_bmp, potrace_input)
    subprocess.run(
        [
            str(POTRACE_PATH),
            "--pgm",
            temp_bmp,
            "-o",
            temp_pgm
        ],
        check=True)
    smoothed = cv2.imread(temp_pgm, cv2.IMREAD_GRAYSCALE)
    _, smoothed = cv2.threshold(
        smoothed,
        127,
        255,
        cv2.THRESH_BINARY_INV)
    height, width = smoothed.shape
    contours = find_contours(smoothed)
    contours = filter_contours(contours)
    contours = simplify_contours(contours)
    paths = []
    for contour in contours:
        path = contour_to_svg_path(contour)
        if path:
            paths.append(path)
    save_svg(
        paths,
        width,
        height,
        output_path)
    os.remove(temp_bmp)
    os.remove(temp_pgm)
    return output_path