import pickle
import os

MODEL_PATH = os.path.join("models", "spam_model.pkl")

def load_model():
    with open(MODEL_PATH, "rb") as f:
        model, tfidf = pickle.load(f)
    return model, tfidf
