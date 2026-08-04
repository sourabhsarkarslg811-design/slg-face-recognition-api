# SLG Face Recognition API

Render-ready FastAPI service for face detection, 512D embeddings, and cosine comparison.

## Deploy
1. Upload all files to GitHub.
2. Render → New Web Service → connect repository.
3. Runtime: Docker, plan: Free.
4. Add environment variable API_SECRET with a long random value.
5. Deploy.

## Endpoints
- GET /health
- POST /warmup
- POST /v1/detect
- POST /v1/detect-url
- POST /v1/compare

## Important licensing note
InsightFace states that its distributed pretrained model packs are for non-commercial research use. Use buffalo_l only for testing unless you have the required commercial rights. For production, replace it with licensed model weights.
