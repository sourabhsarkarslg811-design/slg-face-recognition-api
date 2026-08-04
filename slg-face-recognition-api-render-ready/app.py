from __future__ import annotations
import os
import secrets
from typing import Annotated
import httpx
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field
from face_detector import decode_image, detect_faces, get_face_app
from face_match import compare_embeddings

app = FastAPI(title="SLG Face Recognition API", version="1.0.0")
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(12 * 1024 * 1024)))
API_SECRET = os.getenv("API_SECRET", "")

def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    if not API_SECRET:
        raise HTTPException(status_code=503, detail="API_SECRET is not configured")
    if x_api_key is None or not secrets.compare_digest(x_api_key, API_SECRET):
        raise HTTPException(status_code=401, detail="Invalid API key")

async def read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large")
    return data

class URLRequest(BaseModel):
    image_url: str

class CompareRequest(BaseModel):
    embedding_a: list[float] = Field(min_length=128)
    embedding_b: list[float] = Field(min_length=128)
    threshold: float = Field(default=0.42, ge=-1.0, le=1.0)

@app.get("/")
def root() -> dict[str, str]:
    return {"service": "SLG Face Recognition API", "status": "running"}

@app.get("/health")
def health() -> dict[str, object]:
    return {"ok": True, "model_pack": os.getenv("MODEL_PACK", "buffalo_l"), "api_key_configured": bool(API_SECRET)}

@app.post("/warmup", dependencies=[Depends(require_api_key)])
def warmup() -> dict[str, object]:
    get_face_app()
    return {"ok": True, "message": "Model loaded"}

@app.post("/v1/detect", dependencies=[Depends(require_api_key)])
async def detect(file: Annotated[UploadFile, File(...)]) -> dict[str, object]:
    try:
        faces = detect_faces(decode_image(await read_upload(file)))
        return {"face_count": len(faces), "faces": faces}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.post("/v1/detect-url", dependencies=[Depends(require_api_key)])
async def detect_url(payload: URLRequest) -> dict[str, object]:
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(payload.image_url)
            response.raise_for_status()
        if len(response.content) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Image is too large")
        faces = detect_faces(decode_image(response.content))
        return {"face_count": len(faces), "faces": faces}
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=400, detail=f"Could not download image: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.post("/v1/compare", dependencies=[Depends(require_api_key)])
def compare(payload: CompareRequest) -> dict[str, object]:
    try:
        result = compare_embeddings(payload.embedding_a, payload.embedding_b, payload.threshold)
        return {"matched": result.matched, "similarity": round(result.similarity, 6), "distance": round(result.distance, 6), "threshold": result.threshold}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
