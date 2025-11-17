import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import os

# Create folder if not exists
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("spam_dataset_kaggle_format.csv")   # change if needed

# Rename columns if needed
df.columns = ["label", "text"]

# Convert labels
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Vectorizer
tfidf = TfidfVectorizer(stop_words="english")
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Test
pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, pred))

# Save files
joblib.dump(model, "models/spam_model.pkl")
joblib.dump(tfidf, "models/vectorizer.pkl")

print("Model saved successfully!")
