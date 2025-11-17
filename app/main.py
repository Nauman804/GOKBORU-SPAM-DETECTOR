from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib

app = FastAPI(title="Spam Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("models/spam_model.pkl")
tfidf = joblib.load("models/vectorizer.pkl")

class Message(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "API is running", "usage": "POST /predict"}

@app.post("/predict")
def predict_text(data: Message):
    vector = tfidf.transform([data.text])
    prediction = model.predict(vector)[0]
    label = "spam" if prediction == 1 else "ham"
    return {"text": data.text, "prediction": label}
