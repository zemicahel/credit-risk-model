from pydantic import BaseModel

class CreditScoringRequest(BaseModel):
    # Features required by the trained model (Aggregated features)
    Amount_sum: float
    Amount_mean: float
    Amount_std: float
    TransactionHour_mean: float
    TransactionDay_mean: float
    ProductCategory: str
    Channel: str
    Recency: float
    Frequency: float
    Monetary: float

class CreditScoringResponse(BaseModel):
    is_high_risk: int
    probability: float