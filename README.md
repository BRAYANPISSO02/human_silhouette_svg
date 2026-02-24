# Human Silhouette SVG Generator

## Project Overview
**Human Silhouette SVG Generator** is a project focused on the **automatic extraction of human silhouettes in vector (SVG) format** from real-world images.  
The final objective is to build a **robust and scalable pipeline**, initially based on classical computer vision techniques and later extended with **neural networks for image segmentation**.

In its current state, the project implements a **fully functional baseline** that converts an image of a person into a clean vector silhouette, suitable for:
- laser engraving
- line-art illustration
- geometric analysis
- future machine learning tasks

---

## Current Project Goal (Stage 1)
✔ Convert an RGB image into a **vector human silhouette (SVG)**  
✔ Deterministic and reproducible pipeline  
✔ Solid foundation for dataset generation  

At this stage, **no neural network is used yet**.  
The current approach serves as a **classical baseline** and as an initial data generation tool.

---

## Project Structure

![estructura](F:\proyectos_independientes\human_silhouette_svg\structure\Conversión de foto a silueta SVG.png)

---

## Current Pipeline (Baseline)

The current system flow is as follows:

Input Image (RGB)
→ Foreground Segmentation (GrabCut)
→ Binary Mask Cleaning (Morphology)
→ Main Contour Extraction
→ SVG Path Generation
→ Output Silhouette (SVG)


### 1. Image Loading
- File existence verification
- Image loading using OpenCV

### 2. Foreground Segmentation
- Use of the **GrabCut** algorithm initialized with a rectangular region
- Pixel classification into background / foreground
- Conversion of the result into a binary mask

### 3. Mask Cleaning
- Morphological operations (opening and closing)
- Noise removal and internal hole filling

### 4. Contour Extraction
- External contour detection
- Selection of the main contour based on maximum area

### 5. SVG Vectorization
- Conversion of the contour into SVG commands (`M`, `L`, `Z`)
- Solid silhouette with no stroke (black fill)
- Canvas size based on the original image dimensions

---

## How to Run (Current Version)

### 1. Install dependencies

```bash
pip install -r requirements.txt

### 2. Place input image

Place the input image in:

data/raw/persona_1.jpg

### 3. Run pipeline

python src/pipeline.py

### 4. Output

The result is generated at:

outputs/svg/silhouette.svg

## Current Capabilities

Automatic image → SVG silhouette conversion

Closed and clean silhouettes

Modular and easily extensible pipeline

No dependency on trained models

Useful as an initial generator of masks and contours

## Current Limitations

Sensitive to:

lighting conditions

complex backgrounds

poor framing

Does not generalize well at scale
Does not learn or improve over time
Based on classical heuristics

These limitations are expected and accepted at this stage of the project.

## Planned Next Steps (Stage 2)

Construction of a segmentation dataset

Replacement of GrabCut with a neural network (U-Net / DeepLab)

Geometric normalization of silhouettes

Consistent export for printing, laser engraving, or analysis

Possible human pose classification or analysis

## Long-Term Vision

This project is designed to evolve toward:

deep learning–based segmentation

automatic generation and analysis of human silhouettes

hybrid pipelines (ML + geometry)

industrial and creative applications

## License

Project in experimental / personal phase.
License to be defined.