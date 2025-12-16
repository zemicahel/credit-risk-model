from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from .pydantic_models import CreditScoringRequest, CreditScoringResponse
import os

app = FastAPI(title="Credit Risk Scoring API")

# Load model at startup
MODEL_PATH = "models/best_model.pkl"
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        print("Warning: Model file not found. API will not work correctly.")

@app.post("/predict", response_model=CreditScoringResponse)
def predict(request: CreditScoringRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Convert request to DataFrame
    data_dict = request.dict()
    input_df = pd.DataFrame([data_dict])
    
    try:
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        return CreditScoringResponse(
            is_high_risk=int(prediction),
            probability=float(probability)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}