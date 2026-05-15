"""
app.py — FastAPI Inference Server
E-Scooter Proper/Improper Parking Detector

Usage:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /         → serves index.html
    GET  /health   → health check + model status
    POST /predict  → multipart image → JSON result
"""

import base64
import io
import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = Path(os.getenv("PythonProject5", "./best.pt"))
MAX_FILE_MB = 10
IMGSZ       = int(os.getenv("IMGSZ", "640"))
CONF_THRESH = float(os.getenv("CONF_THRESH", "0.25"))
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Scooter Parking Detector", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = None

def load_model():
    global model
    if not MODEL_PATH.exists():
        logger.warning(f"Model not found at '{MODEL_PATH}'. Train first with train.py.")
        return
    try:
        from ultralytics import YOLO
        logger.info(f"Loading model: {MODEL_PATH}")
        model = YOLO(str(MODEL_PATH))
        logger.info("Model loaded.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Код, который выполнится при СТАРТЕ
    load_model()
    yield
    # Код, который выполнится при ВЫКЛЮЧЕНИИ (если нужно)
    logger.info("Shutting down...")

app = FastAPI(title="Scooter Parking Detector", version="1.0.0", lifespan=lifespan)

# ── Helpers ───────────────────────────────────────────────────────────────────
CLASS_ADVICE = {
    "proper":   "The scooter appears to be parked correctly — upright, out of pedestrian paths, and in a designated area.",
    "improper": "The scooter appears to be parked incorrectly. It may be blocking a path, lying on its side, or parked in a restricted zone.",
}

CLASS_DISPLAY = {
    "proper":   "Properly Parked ✅",
    "improper": "Improperly Parked ❌",
}

def pil_to_base64(img: Image.Image, fmt="JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def run_inference(image: Image.Image) -> dict:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train.py first, then restart the server."
        )

    t0 = time.perf_counter()
    results = model.predict(source=image, imgsz=IMGSZ, conf=CONF_THRESH, verbose=False)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    result = results[0]
    boxes  = result.boxes
    names  = result.names  # {0: 'improper', 1: 'proper'}

    # ── Aggregate detections ──────────────────────────────────────────────────
    # Tally max confidence per class across all detections
    class_conf: dict[str, float] = {}
    detections = []

    if boxes is not None and len(boxes):
        for box in boxes:
            cls_id   = int(box.cls.item())
            conf     = float(box.conf.item())
            cls_name = names.get(cls_id, f"class_{cls_id}")
            xyxy     = box.xyxy[0].cpu().numpy().tolist()  # [x1,y1,x2,y2]

            detections.append({
                "class":      cls_name,
                "confidence": round(conf * 100, 1),
                "bbox":       [round(v, 1) for v in xyxy],
            })
            if conf > class_conf.get(cls_name, 0.0):
                class_conf[cls_name] = conf

    # ── Determine verdict ─────────────────────────────────────────────────────
    if not class_conf:
        # No detections at all
        verdict     = "improper"
        top_conf    = 0.0
        no_detection = True
    else:
        verdict      = max(class_conf, key=class_conf.get)
        top_conf     = class_conf[verdict]
        no_detection = False

    # ── Draw bounding boxes on image ──────────────────────────────────────────
    annotated_b64 = None
    try:
        annotated_arr = result.plot()          # numpy BGR array with boxes drawn
        annotated_pil = Image.fromarray(annotated_arr[..., ::-1])  # BGR→RGB
        annotated_b64 = pil_to_base64(annotated_pil)
    except Exception as e:
        logger.warning(f"Could not render annotated image: {e}")

    return {
        "label":        verdict,
        "display":      CLASS_DISPLAY[verdict],
        "confidence":   round(top_conf * 100, 1),
        "advice":       CLASS_ADVICE[verdict] if not no_detection
                        else "No scooter detected in the image. Please upload a clearer photo.",
        "no_detection": no_detection,
        "detections":   detections,
        "annotated_image": annotated_b64,   # base64 JPEG, drawn by ultralytics
        "inference_ms": elapsed_ms,
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": model is not None,
        "model_path":   str(MODEL_PATH),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(415, f"Unsupported file type '{file.content_type}'.")

    raw = await file.read()
    if len(raw) / 1024 / 1024 > MAX_FILE_MB:
        raise HTTPException(413, f"File too large. Max {MAX_FILE_MB} MB.")

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Could not decode image: {e}")

    logger.info(f"Predicting: {image.size}, {file.content_type}, {len(raw)/1024:.1f} KB")
    result = run_inference(image)
    logger.info(f"→ {result['label']} ({result['confidence']}%) in {result['inference_ms']} ms")

    return JSONResponse(content=result)


@app.get("/")
def serve_frontend():
    p = Path("index.html")
    if p.exists():
        return FileResponse("index.html")
    return {"message": "Place index.html alongside app.py to serve the frontend."}


if __name__ == "__main__":
    import uvicorn
    # Запускаем сервер на порту 8000
    uvicorn.run(app, host="localhost", port=8000)