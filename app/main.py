from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
from fastapi.responses import FileResponse


app = FastAPI(title="Spam Detection API")
@app.get("/")
def home():
    return FileResponse("index.html")
# CORS Settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model and Vectorizer
model = joblib.load("models/spam_model.pkl")
tfidf = joblib.load("models/vectorizer.pkl")


# Home Route (IMPORTANT)
@app.get("/")
def home():
    return {"message": "Spam Detector API is running!"}


# Request Body
class Message(BaseModel):
    text: str


# Prediction API
@app.post("/predict")
def predict_text(data: Message):
    vector = tfidf.transform([data.text])
    prediction = model.predict(vector)[0]
    label = "spam" if prediction == 1 else "ham"
    return {"text": data.text, "prediction": label}



