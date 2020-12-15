# Library import
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from classifier import Classifier
from helper import SentimentRequest, SentimentResponse

# Create APP intance of FastAPI
app = FastAPI()
model = Classifier()
templates = Jinja2Templates(directory="templates")

# Index route. Default: http://127.0.0.1:8000
@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    context = {
        "request" : request,
        'title' : "Form Input for News Classifier"
    }
    return templates.TemplateResponse("index.html", context=context)

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