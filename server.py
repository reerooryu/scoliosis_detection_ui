# Inference server: runs the vertebra model on an uploaded X-ray and returns
# the result as JSON. Endpoints: POST /predict (an image file), GET /health.
# The model itself is in modules/inference.py, which the app's Local mode
# uses too.

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from modules.inference import DEFAULT_WEIGHTS_PATH, load_predictor, run_inference

WEIGHTS_PATH = DEFAULT_WEIGHTS_PATH

app = FastAPI(title="Cobb Angle Inference API")


@app.post("/predict")
def predict_api(file: UploadFile = File(...)):
    if (file.content_type or "").split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = file.file.read()
    try:
        result = run_inference(image_bytes, load_predictor(WEIGHTS_PATH))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return JSONResponse(content=result)


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    # Load the model before opening the port, so a missing weights file
    # fails at startup instead of on the first request.
    load_predictor(WEIGHTS_PATH)
    uvicorn.run(app, host="0.0.0.0", port=4000)
