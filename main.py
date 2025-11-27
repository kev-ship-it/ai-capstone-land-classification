from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from keras_vit_loader import predict_keras_vit

app = FastAPI(title="AI Capstone – Land Classification (Keras ViT)")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        return JSONResponse(status_code=400, content={"error": "No file uploaded"})
    try:
        p_agri, label = predict_keras_vit(file.file)
        return {
            "framework": "keras",
            "filename": file.filename,
            "prediction": label,
            "prob_agri": float(p_agri),
            "prob_non_agri": float(1.0 - p_agri),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
