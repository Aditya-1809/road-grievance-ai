from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Create FastAPI app
app = FastAPI()

# Load dataset
df = pd.read_csv("roads.csv")

# Features and target
X = df["complaint_text"]
y = df["sentiment"]

# Convert text to vectors
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vectorized, y)

# Input structure
class Complaint(BaseModel):
    complaint: str

# Prediction API
@app.post("/predict")
def predict(data: Complaint):

    text = vectorizer.transform([data.complaint])
    prediction = model.predict(text)

    return {
        "complaint": data.complaint,
        "predicted_sentiment": prediction[0]
    }