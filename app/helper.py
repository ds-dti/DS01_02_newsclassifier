from pydantic import BaseModel

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    text: str
    prediction: dict
