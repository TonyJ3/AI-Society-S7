from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("hit_predictor_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define expected input features
class SongFeatures(BaseModel):
    energy: float
    tempo: float
    danceability: float
    loudness: float
    liveness: float
    valence: float
    time_signature: int
    speechiness: float
    instrumentalness: float
    mode: int
    key: int
    duration_ms: int
    acousticness: float

app = FastAPI()

@app.post("/predict")
def predict(features: SongFeatures):
    # Create input data for array
    input_data = np.array([[features.energy, features.tempo, features.danceability,
                            features.loudness, features.liveness, features.valence,
                            features.time_signature,features.speechiness, 
                            features.instrumentalness,features.mode,
                            features.key, features.duration_ms, features.acousticness]])
    
    # input the data into the scaler
    scaled_input = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_input)[0]
    label = "Hit" if prediction == 1 else "Not a Hit"
    return {"prediction": label}