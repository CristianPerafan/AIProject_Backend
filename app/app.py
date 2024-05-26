from fastapi import FastAPI, File, UploadFile

from model.model import __version__ as model_version
from model.model import predict_pipeline


app = FastAPI()


@app.get("/")
def home():
    return {
        'message': 'Welcome to the API',
        'version': model_version
    }

@app.post("/predict/")
def predict(file: UploadFile = File(...)):
    return predict_pipeline(file.file)