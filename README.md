Citizen Grievance AI - Roads Department

Project Overview

Citizen Grievance AI is an NLP-based complaint classification system developed for the Roads Department. The system analyzes citizen road-related complaints and predicts the sentiment category of road issues using Machine Learning and Natural Language Processing (NLP).

The model classifies complaints into:

- Critical
- Negative
- Neutral
- Positive

Example

Input:

Huge potholes near station causing accidents

Output:

Critical

---

Features

✔ NLP-based complaint sentiment analysis
✔ TF-IDF text vectorization
✔ Logistic Regression classification model
✔ Complaint sentiment prediction system
✔ Exploratory Data Analysis (EDA)
✔ Confusion Matrix visualization
✔ Word Cloud generation
✔ FastAPI integration for real-time prediction
✔ Swagger API documentation ("/docs")

---

Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook
- FastAPI
- Uvicorn
- VS Code
- Git & GitHub

---

Dataset

The dataset contains road-related citizen complaints categorized into sentiment labels:

- Critical
- Negative
- Neutral
- Positive

Dataset File:
"roads.csv"

---

Project Structure

citizen-grievance-ai
│── roads.csv
│── roads_eda.ipynb
│── train_model.py
│── predict.py
│── api.py
│── clean_text.py
│── README.md

---

Model Used

TF-IDF Vectorization

Used to convert complaint text into numerical form for machine learning.

Logistic Regression

Used as the classification algorithm for sentiment prediction.

---

Model Accuracy

Achieved approximately:

80% accuracy

on the Roads grievance dataset.

---

How to Run the Project

1. Install Dependencies

pip install pandas scikit-learn matplotlib jupyter notebook fastapi uvicorn

2. Train Model

python train_model.py

3. Predict Complaint Sentiment

python predict.py

---

FastAPI Integration

Run API Server

uvicorn api:app --reload

Open API Documentation

http://127.0.0.1:8000/docs

The Swagger UI allows testing complaint predictions directly through the browser.

Sample API Request

{
  "complaint": "Huge pothole near railway station"
}

Sample API Response

{
  "complaint": "Huge pothole near railway station",
  "predicted_sentiment": "Critical"
}

---

Sample Prediction

Input:

Road repaired successfully

Output:

Positive

---

Author

Aditya Dhiraj Bhandarkar