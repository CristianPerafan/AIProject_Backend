from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from model.model import __version__ as model_version
from model.model import predict_pipeline
from pydub import AudioSegment
import os

app = FastAPI()


origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        'message': 'Welcome to the API',
        'version': model_version
    }

@app.post("/predict/")
def predict(file: UploadFile = File(...)):


    try:
        with open(f"audio/{file.filename}", "wb") as f:
            f.write(file.file.read())

        prediction = predict_pipeline(f"audio/{file.filename}")

        os.remove(f"audio/{file.filename}")

        return prediction[0]
    except:
        return {
            'message': 'An error occurred'
        }
    
