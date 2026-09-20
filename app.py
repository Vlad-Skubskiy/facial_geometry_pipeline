import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import RedirectResponse
from geometry import FacialGeometryPipeline
import cv2

app = FastAPI()
model = FacialGeometryPipeline()

@app.get("/")
def main_page():
    return RedirectResponse(url="/docs")

@app.get('/health')
def health_check():
    return {"status": "ok", "model": "MediaPipe FaceMesh loaded"}


@app.post('/process-face')
async def process_face(file: UploadFile = File(...)):

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    result = model.process_image(image_np)

    if image_np is None:
        raise HTTPException(status_code=400, detail="...")

    if 'error' in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result