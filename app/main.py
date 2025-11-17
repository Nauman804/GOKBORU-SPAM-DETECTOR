from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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

# load ML model
model = joblib.load("app/models/spam_model.pkl")
tfidf = joblib.load("app/models/vectorizer.pkl")


class Message(BaseModel):
    text: str

@app.get("/")
def home():
    return FileResponse("app/static/index.html")

@app.post("/predict")
def predict_text(data: Message):
    vector = tfidf.transform([data.text])
    prediction = model.predict(vector)[0]
    label = "spam" if prediction == 1 else "ham"
    return {"text": data.text, "prediction": label}

# serve CSS, JS, images if any
app.mount("/static", StaticFiles(directory="app/static"), name="static")
