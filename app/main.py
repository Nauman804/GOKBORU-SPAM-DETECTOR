from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib

app = FastAPI(title="Spam Detection API")

# CORS Settings (allow frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------ Load Model & Vectorizer ------------
# Make sure these files exist inside /models/
model = joblib.load("models/spam_model.pkl")
tfidf = joblib.load("models/vectorizer.pkl")


# ------------ Request Body ------------
class Message(BaseModel):
    text: str


# ------------ Prediction API ------------
@app.post("/predict")
def predict_text(data: Message):
    vector = tfidf.transform([data.text])
    prediction = model.predict(vector)[0]
    label = "spam" if prediction == 1 else "ham"
    return {"text": data.text, "prediction": label}


