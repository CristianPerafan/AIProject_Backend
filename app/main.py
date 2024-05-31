from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from app.model.model import __version__ as model_version
from app.model.model import predict_pipeline
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

        os.makedirs('audio', exist_ok=True)

        with open(f"audio/{file.filename}", "wb") as f:
            f.write(file.file.read())

        prediction = predict_pipeline(f"audio/{file.filename}")

        os.remove(f"audio/{file.filename}")

        return prediction[0]
    except Exception as e:
        print(f"An error occurred: {e}")
        return {
            'message': 'An error occurred',
            'error': str(e)
        }

    
