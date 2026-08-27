import sys
from pathlib import Path
import cv2
import torch
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse

# 1. ENSURE PYTHON FINDS THE PROJECT ROOT
# ------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FRONTEND_DIR
from src.main import load_unet_model, run_pipeline
from src.preprocessing import load_sam

# 2. INITIALIZE FASTAPI APPLICATION
# ------------------------------------------------------------------------------
app = FastAPI(
    title="AI Vectorizer Pro API",
    description="Backend to convert photos of people into SVG vector silhouettes",
    version="1.0.0"
)

# 3. LOAD MODELS INTO MEMORY (ONCE AT STARTUP)
# ------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Initializing server on device: {device}")

model = load_unet_model(device)
print("[INFO] U-Net model loaded successfully.")

predictor = load_sam()
print("[INFO] SAM model loaded successfully.")


# 4. API ENDPOINTS
# ------------------------------------------------------------------------------
@app.post("/api/vectorize")
async def vectorize_image(file: UploadFile = File(...)):
    """
    Receives an uploaded image from the web and runs the central pipeline automatically.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo subido no es una imagen válida.")

    try:
        # Read image bytes and decode to OpenCV BGR array
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image_bgr is None:
            raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen.")

        # Execute the unified pipeline in automatic mode
        svg_path, svg_content = run_pipeline(
            image_bgr,
            model=model,
            predictor=predictor,
            interactive=False,
            device=device
        )

        return JSONResponse({
            "status": "success",
            "filename": Path(svg_path).name,
            "svg": svg_content
        })

    except Exception as e:
        print(f"[ERROR] Failed to process image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 5. SERVE FRONTEND STATIC ASSETS
# ------------------------------------------------------------------------------
@app.get("/")
async def serve_index():
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html no encontrado en frontend/")
    return FileResponse(str(index_file))

@app.get("/app.js")
async def serve_app_js():
    js_file = FRONTEND_DIR / "app.js"
    if not js_file.exists():
        raise HTTPException(status_code=404, detail="app.js no encontrado en frontend/")
    return FileResponse(str(js_file), media_type="application/javascript")


# 6. DIRECT SERVER EXECUTION
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)
