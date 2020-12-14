# Library import
import uvicorn
from fastapi import FastAPI
from classifier import Classifier
from helper import SentimentRequest, SentimentResponse

# Create APP intance of FastAPI
app = FastAPI()
model = Classifier()

# Index route. Default: http://127.0.0.1:8000
@app.post('/predict/', response_model=SentimentResponse, status_code=200)
async def predict_text(request: SentimentRequest):

    if not model:
        raise HTTPException(status_code=404, detail="Model not found.")

    pred = model.process(request.text)

    return SentimentResponse(text=request.text, prediction=pred)

# Run the API with uvicorn
# API will run on http://127.0.0.1:8000 by default
if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)