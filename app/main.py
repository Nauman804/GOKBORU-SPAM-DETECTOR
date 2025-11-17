from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import joblib

app = FastAPI(title="Spam Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (MUST be above routes)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Load model
model = joblib.load("app/models/spam_model.pkl")
tfidf = joblib.load("app/models/vectorizer.pkl")

class Message(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
def home():
    with open("app/static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
def predict_text(data: Message):
    vector = tfidf.transform([data.text])
    prediction = model.predict(vector)[0]
    label = "spam" if prediction == 1 else "ham"
    return {"text": data.text, "prediction": label}
