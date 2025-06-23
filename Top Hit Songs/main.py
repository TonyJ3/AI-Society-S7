from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
def predict(features: SongFeatures):

    # Convert input to a DataFrame with correct column names
    input_data = pd.DataFrame([features.model_dump()])
    
    # input the data into the scaler
    scaled_input = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_input)[0]
    label = "Hit" if prediction == 1 else "Not a Hit"
    return {"prediction": label}