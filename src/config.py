from pathlib import Path

# ==============================================================================
# BASE PROJECT PATHS
# ==============================================================================
# PROJECT_ROOT points dynamically to the root folder of the project
# (one level above this 'src' directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Core project directories
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
BIN_DIR = PROJECT_ROOT / "bin"

# ==============================================================================
# DATASET DIRECTORIES & FILES
# ==============================================================================
TRAIN_INPUT_DIR = DATA_DIR / "input_train"
TRAIN_OUTPUT_DIR = DATA_DIR / "output_train"
VALIDATION_INPUT_DIR = DATA_DIR / "input_val"
VALIDATION_OUTPUT_DIR = DATA_DIR / "output_val"

INPUT_SEG_DIR = DATA_DIR / "input_segmentation"
OUTPUT_SEG_DIR = DATA_DIR / "output_segmentation"
STATE_FILE = DATA_DIR / "preprocessing_state.json"

# ==============================================================================
# MODEL CHECKPOINTS & BINARIES
# ==============================================================================
BEST_MODEL_PATH = CHECKPOINTS_DIR / "best_model.pth"
FINAL_MODEL_PATH = CHECKPOINTS_DIR / "model_final.pth"
SAM_CHECKPOINT = PROJECT_ROOT / "segment_anything" / "sam_vit_h_4b8939.pth"
POTRACE_PATH = BIN_DIR / "potrace.exe"

# ==============================================================================
# OUTPUT DIRECTORIES
# ==============================================================================
OUTPUT_PREPROCESSING_DIR = OUTPUTS_DIR / "preprocessing"
OUTPUT_UNET_DIR = OUTPUTS_DIR / "U_Net"
OUTPUT_POSTPROCESSING_DIR = OUTPUTS_DIR / "postprocessing"
OUTPUT_VECTORIZATION_DIR = OUTPUTS_DIR / "vectorization"

# Helper function to ensure all required output directories exist
def ensure_output_dirs():
    for output_dir in [
        OUTPUT_PREPROCESSING_DIR,
        OUTPUT_UNET_DIR,
        OUTPUT_POSTPROCESSING_DIR,
        OUTPUT_VECTORIZATION_DIR
    ]:
        output_dir.mkdir(parents=True, exist_ok=True)

