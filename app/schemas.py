from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)

class PredictionResponse(BaseModel):
    prediction: int
    class_name: str
    confidence: float
