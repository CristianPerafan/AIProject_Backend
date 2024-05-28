
from pathlib import Path
import librosa
import numpy as np
import pandas as pd
import joblib


"""
    version: 0.1.0 - Includes unbalanced classes distribution
"""


__version__ = '0.1.0'

BASE_DIR = Path(__file__).resolve(strict=True).parent


with open(f"{BASE_DIR}/knn-{__version__}.pkl","rb") as f:
    knn = joblib.load(f)


with open(f"{BASE_DIR}/pca-{__version__}.pkl","rb") as f:
    pca = joblib.load(f)


RATE_HZ = 16000 # Audios are being sampled at 16,000 times per second

def feature_extraction_with_means(file_path):


    features = {}

    audio,_ = librosa.load(file_path) # Load the audio file (sr=None to keep the original sampling rate
    
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=RATE_HZ)) # Compute the spectral centroid
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=RATE_HZ)) # Compute the spectral bandwidth
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=RATE_HZ)) # Compute the spectral rolloff
    features['spectral_centroid'] = spectral_centroid
    features['spectral_bandwidth'] = spectral_bandwidth
    features['spectral_rolloff'] = spectral_rolloff
    
    mfcc = librosa.feature.mfcc(y=audio, sr=RATE_HZ) # Compute the mel-frequency cepstral coefficients
    for i, el in enumerate(mfcc):
        features[f'mfcc_{i+1}'] = np.mean(el)
    
    return pd.DataFrame([features])
    
    


def predict_pipeline(file_path):
    # Extraer características
    features = feature_extraction_with_means(file_path)
    
    # Convertir DataFrame a numpy array si es necesario
    if isinstance(features, pd.DataFrame):
        features = features.to_numpy()
    
    # Transformar características usando PCA
    scaled_features = pca.transform(features)
    
    # Crear DataFrame con las características escaladas
    scaled_features_df = pd.DataFrame(data=scaled_features, 
                                      columns=[f'pc_{i}' for i in range(1, 16)])
    
    # Obtener las clases que el modelo puede predecir
    class_labels = knn.classes_
    print(class_labels)
    
    # Obtener las probabilidades de predicción
    results = knn.predict_proba(scaled_features_df)

    print(knn.predict(scaled_features_df))
    
    # Formar una lista de diccionarios con etiquetas y probabilidades
    predictions = []
    for sample_probs in results:
        sample_prediction = []
        for label, prob in zip(class_labels, sample_probs):
            sample_prediction.append({'label': label, 'probability': prob})
        predictions.append(sample_prediction)

    
    
    return predictions
    