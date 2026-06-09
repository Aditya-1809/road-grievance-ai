from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Create FastAPI app
app = FastAPI()

# Load saved model
grievance_classifier = joblib.load(
    "grievance_model.pkl"
)

# Load saved vectorizer
text_encoder = joblib.load(
    "text_encoder.pkl"
)

# Input structure
class Complaint(BaseModel):
    complaint: str


# Prediction API
@app.post("/predict")
def predict(data: Complaint):

    processed_text = text_encoder.transform(
        [data.complaint]
    )

    prediction = grievance_classifier.predict(
        processed_text
    )

    return {
        "complaint": data.complaint,
        "predicted_sentiment": prediction[0]
    }