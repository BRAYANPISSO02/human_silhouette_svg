# Human Silhouette SVG

A computer vision pipeline that converts photos of people into clean **SVG** vector silhouettes suitable for laser cutting/engraving, plotting, or vector illustration.

Standard edge detectors (Canny/Sobel) and running Potrace directly over raw photographs break down quickly on real-world portraits: fabric textures, shadows, and cluttered backgrounds produce thousands of noisy paths. This project splits the problem into four stages: foreground person segmentation with **SAM (ViT-H)**, image-to-line-art translation with a custom **U-Net** trained using L1 + VGG-19 perceptual loss, classical morphological cleanup with thinning, and final SVG path export.

---

## Pipeline architecture

The end-to-end flow lives in `src/main.py` (`run_pipeline`) and is shared by both the interactive desktop tool and the FastAPI web service:

```text
Input RGB Photo (file path or in-memory buffer)
   │
   ▼
1. Preprocessing (src/preprocessing.py)
   ├─ Subject segmentation with SAM (ViT-H): interactive click mode or automatic mode
   ├─ Background removal via binary mask (bitwise AND)
   ├─ Aspect-ratio-preserving letterbox to 512x512
   └─ Tensor normalization [3, 512, 512] in [-1, 1]
   │
   ▼
2. Line-Art Inference (UNet in src/training.py)
   ├─ Symmetric Encoder-Decoder (64 → 128 → 256 → 512 → 1024) with skip connections
   └─ Continuous single-channel output [1, 512, 512] in [-1, 1] (Tanh)
   │
   ▼
3. Postprocessing (src/postprocessing.py)
   ├─ Percentile contrast stretching (p1 – p99)
   ├─ Gaussian adaptive thresholding (block_size=31, C=-5)
   ├─ Connected-component noise filtering (area >= 25 px)
   ├─ Elliptical morphological closing (3x3)
   └─ Skeletonization / thinning (cv2.ximgproc.thinning)
   │
   ▼
4. Vectorization (src/vectorization.py)
   ├─ Raster smoothing pass via Potrace (--pgm)
   ├─ Contour extraction (cv2.findContours) and area filtering (area >= 8.0)
   ├─ Douglas-Peucker polygonal simplification (epsilon = 0.0015 * perimeter)
   └─ SVG serialization (<svg><path d="M ... L ... Z"/></svg>)
```

---

## Key design choices

- **Decoupling segmentation from line-art generation (SAM + U-Net):** training a U-Net directly on unsegmented photos requires a much larger dataset just to teach the model to ignore arbitrary backgrounds. Using SAM ViT-H up front strips background clutter so the U-Net only has to learn the mapping from an isolated person to clean strokes, making training viable with a small custom dataset (72 paired training samples).
- **512x512 letterboxing instead of direct resizing:** preserving aspect ratio with black padding avoids stretching or squashing body proportions across portrait and landscape inputs.
- **Combined L1 + Perceptual Loss (`L1 + 0.1 * PerceptualLoss`):** pixel-wise L1 or MSE alone tends to blur or break thin strokes under slight spatial misalignment. Adding a perceptual term computed over the first 16 frozen layers of VGG-19 (`IMAGENET1K_V1` weights) encourages structural continuity along contours.
- **Classical CV postprocessing before vectorization:** U-Net predicts a continuous grayscale map in `[-1, 1]`, not a clean 1-pixel binary stroke map. Percentile normalization, adaptive thresholding, area filtering, and thinning normalize line width before feeding Potrace and `approxPolyDP`.
- **Custom labeling utility (`src/dataset_builder.py`):** since no off-the-shelf dataset matched the target line-art style, a Tkinter + OpenCV + SAM helper script was built to curate aligned `(input, target)` pairs while persisting labeling progress in `data/preprocessing_state.json`.

---

## Model & training setup

All training components live in `src/training.py`.

### U-Net architecture

| Stage | Operations | Channels | Spatial size |
| :--- | :--- | :---: | :---: |
| Encoder (x4) | `2x (Conv3x3 → BatchNorm → ReLU)` + `MaxPool2d(2)` | `3 → 64 → 128 → 256 → 512` | `512 → 32` |
| Bottleneck | `2x (Conv3x3 → BatchNorm → ReLU)` | `512 → 1024` | `32x32` |
| Decoder (x4) | `ConvTranspose2d(2)` + `concat(skip)` + `ConvBlock` | `1024 → 512 → 256 → 128 → 64` | `32 → 512` |
| Output head | `Conv1x1` + `Tanh` | `64 → 1` | `512x512` |

The trained `state_dict` (`checkpoints/best_model.pth`) is ~142 MB in `float32`.

### Training configuration

| Parameter | Value |
| :--- | :--- |
| Input / Output resolution | `512x512` RGB → `512x512` grayscale |
| Normalization | `[-1, 1]` (`mean=0.5, std=0.5`) |
| Loss function | `L1Loss + 0.1 * PerceptualLoss(VGG19[:16])` |
| Optimizer | `Adam` (`lr = 1e-3`) |
| Scheduler | `ReduceLROnPlateau(mode="min", factor=0.1, patience=10)` |
| Batch size / Max epochs | `8` / `500` |
| Early stopping | `patience = 5` on validation loss |
| Checkpointing | Every `5` epochs (`checkpoints/checkpoint_epoch_<N>.pth`) + `best_model.pth` |

### Data augmentation

Applied synchronously to `(image, mask)` pairs with `albumentations` on the training split only:
- Random rotation (`±10°`, `p=0.5`).
- Affine scale `(0.9, 1.1)` and translation `(-5%, +5%)` (`p=0.5`).
- Random brightness/contrast adjustment (`±15%`, `p=0.5`).

Training automatically resumes from the highest-numbered epoch checkpoint found in `checkpoints/`.

---

## Repository layout

```text
human_silhouette_svg/
├── bin/
│   └── potrace.exe               # Potrace binary for Windows
├── checkpoints/                  # U-Net weights (best_model.pth, model_final.pth, periodic checkpoints)
├── data/
│   ├── input_train/              # Segmented 512x512 RGB training images (72 samples)
│   ├── output_train/             # Target grayscale line-art images (72 samples)
│   ├── input_val/                # Validation input images
│   ├── output_val/               # Validation target images (required to run training.py)
│   ├── input_segmentation/       # Raw photos used by dataset_builder.py
│   └── output_segmentation/      # Raw targets used by dataset_builder.py
├── frontend/
│   ├── index.html                # Web UI (Tailwind CSS via CDN)
│   └── app.js                    # Drag-and-drop upload, /api/vectorize call, SVG download
├── model_u2net/                  # Artifacts from earlier segmentation experiments (unused in pipeline)
├── outputs/                      # Intermediate debug outputs (preprocessing, U_Net, postprocessing) & SVGs
├── segment_anything/             # Meta AI SAM source + sam_vit_h_4b8939.pth checkpoint
├── src/
│   ├── api.py                    # FastAPI backend and static frontend serving
│   ├── config.py                 # Centralized paths and Potrace binary resolution
│   ├── dataset_builder.py        # GUI helper to build paired training samples with SAM
│   ├── main.py                   # Unified pipeline (run_pipeline) and desktop entry point
│   ├── postprocessing.py         # Contrast stretching, adaptive thresholding, cleanup, thinning
│   ├── preprocessing.py          # SAM loader, interactive/auto segmentation, letterboxing
│   ├── training.py               # Dataset, U-Net model, CombinedLoss, and training loop
│   ├── utils.py                  # Input/output tensor transforms and denormalization
│   └── vectorization.py          # Potrace integration, contour extraction, SVG writer
├── yolov8n-seg.pt                # Weights from earlier segmentation experiments (unused in pipeline)
└── requirements.txt
```

> Note: `model_u2net/` and `yolov8n-seg.pt` are left over from early segmentation benchmarks before standardizing on SAM ViT-H; the active pipeline does not import them.

---

## Setup

### Prerequisites

- **Python:** 3.10 – 3.12 (recommended for PyTorch and `opencv-contrib-python` compatibility).
- **GPU (optional, recommended for training):** NVIDIA GPU with CUDA 12.1 support.
- **Potrace:**
  - On Windows, `bin/potrace.exe` is used automatically.
  - On Linux/macOS, install `potrace` on your `PATH` (`sudo apt install potrace` / `brew install potrace`); `src/config.py` resolves it automatically via `shutil.which("potrace")`.

### Dependencies

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

If you plan to run the web service (`src/api.py`), install the server dependencies that are not currently pinned in `requirements.txt`:

```bash
pip install fastapi uvicorn python-multipart pillow
```

### Required weights

Make sure both weight files are in place before running inference:

1. **SAM ViT-H:** `segment_anything/sam_vit_h_4b8939.pth` (available from the official [Segment Anything](https://github.com/facebookresearch/segment-anything#model-checkpoints) repository).
2. **Trained U-Net:** `checkpoints/best_model.pth` (produced by `src/training.py`).

---

## Usage

### 1. Web UI & REST API

Start the FastAPI server (loads U-Net and SAM once at startup):

```bash
python src/api.py
```

Or run Uvicorn directly:

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

- Web UI: `http://127.0.0.1:8000`
- OpenAPI docs (Swagger): `http://127.0.0.1:8000/docs`

Direct API call example:

```bash
curl -X POST "http://127.0.0.1:8000/api/vectorize" \
  -F "file=@path/to/photo.jpg"
```

Response payload:

```json
{
  "status": "success",
  "filename": "silhouette_a1b2c3d4.svg",
  "svg": "<svg xmlns=\"http://www.w3.org/2000/svg\" ...>...</svg>"
}
```

### 2. Desktop mode (interactive segmentation)

Recommended when photos contain multiple people or cluttered backgrounds where automatic point prompts fall short:

```bash
python src/main.py
```

1. Pick an image in the Tkinter file dialog.
2. In the OpenCV window, **left-click** on regions of the person to accumulate SAM masks:
   - `Z`: undo last mask.
   - `S`: confirm mask and continue the pipeline.
   - `ESC`: cancel.
3. The final SVG is written to `outputs/vectorization/<name>.svg`, and intermediate debug outputs are saved under `outputs/{preprocessing,U_Net,postprocessing}/`.

### 3. Building training pairs

```bash
python src/dataset_builder.py
```

Saves aligned pairs into `data/input_train/` and `data/output_train/` with incremental filenames (`imagen_1.png`, `imagen_2.png`, ...) while tracking the last processed index in `data/preprocessing_state.json`.

### 4. Training U-Net

Ensure matching `.png` pairs exist in `data/input_train/`, `data/output_train/`, `data/input_val/`, and `data/output_val/` (`input_val` should contain only valid `.png` files with matching targets in `output_val`):

```bash
python src/training.py
```

---

## Known limitations & next steps

Concrete issues observed in the current codebase to address in upcoming iterations:

- **Model/training coupling:** `UNet` is defined inside `src/training.py`, forcing `src/main.py` to pull in training-only dependencies (`albumentations`, `vgg19`) at inference time. Extracting `UNet` and its building blocks into `src/model.py` will cleanly decouple inference from training.
- **Scheduler vs. early stopping conflict:** `EARLY_STOPPING_PATIENCE = 5` stops training before `ReduceLROnPlateau(patience=10)` can ever decay the learning rate. These should be inverted (e.g., `scheduler.patience = 4`, `early_stopping = 10`).
- **SAM device placement and automatic prompts:** `load_sam()` in `src/preprocessing.py` does not move SAM onto CUDA (`sam.to(device)`), so it runs on CPU even on GPU machines. Additionally, automatic mode uses three hardcoded vertical guide points assuming a centered single person; replacing those heuristics with a person detector (e.g., YOLOv8 bounding box fed as a box prompt into SAM) would handle off-center and multi-person photos far better.
- **Vector quality and concurrency in `src/vectorization.py`:** Potrace is currently used only as a raster smoother (`--pgm`), and the exported SVG uses polygonal line segments (`M`, `L`, `Z`). Switching to Potrace's native cubic Bézier SVG output (`potrace -s`) would cut node count and produce smoother curves for laser cutting. Temp files should also use `tempfile` to avoid filename collisions under concurrent API requests.
- **Dataset size and evaluation:** 72 training pairs limit generalization to unusual poses and occlusions. Expanding the dataset, fixing random seeds, adding quantitative validation metrics (IoU / Dice on binarized strokes), and pinning FastAPI dependencies in `requirements.txt` are the immediate priorities.
