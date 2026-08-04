from __future__ import annotations
import os
import threading
from functools import lru_cache
from typing import Any
import cv2
import numpy as np
from insightface.app import FaceAnalysis

_MODEL_LOCK = threading.Lock()

@lru_cache(maxsize=1)
def get_face_app() -> FaceAnalysis:
    model_name = os.getenv("MODEL_PACK", "buffalo_l")
    ctx_id = int(os.getenv("CTX_ID", "-1"))
    det_size = int(os.getenv("DET_SIZE", "640"))
    with _MODEL_LOCK:
        app = FaceAnalysis(
            name=model_name,
            providers=["CPUExecutionProvider"],
            root=os.getenv("INSIGHTFACE_ROOT", "/tmp/.insightface"),
        )
        app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
        return app

def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid or unsupported image")
    return image

def detect_faces(image: np.ndarray) -> list[dict[str, Any]]:
    faces = get_face_app().get(image)
    results: list[dict[str, Any]] = []
    for face in faces:
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            continue
        results.append({
            "bbox": [float(v) for v in face.bbox.tolist()],
            "detection_score": float(face.det_score),
            "embedding": [float(v) for v in embedding.tolist()],
        })
    return results
