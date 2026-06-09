Citizen Grievance AI – Roads Department

Project Overview

Citizen Grievance AI is a Machine Learning–based complaint classification system developed for analyzing road-related citizen grievances. The objective of this project is to help local authorities identify the severity of road complaints and improve response prioritization.

The system processes road complaints submitted by citizens and predicts the complaint category using Natural Language Processing (NLP) and Machine Learning techniques.

This project was inspired by real-world civic problems related to road maintenance and public infrastructure, where complaints such as potholes, damaged roads, waterlogging, and unsafe travel conditions often require faster attention.

The model classifies complaints into four categories:

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

Key Features

✔ Complaint classification using Machine Learning
✔ NLP-based text preprocessing and feature extraction
✔ TF-IDF vectorization for complaint analysis
✔ Logistic Regression model for prediction
✔ Exploratory Data Analysis (EDA) using Jupyter Notebook
✔ FastAPI integration for real-time complaint prediction
✔ Interactive Swagger API documentation ("/docs")
✔ Trained model saving using Joblib (".pkl" files)

---

Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Uvicorn
- Matplotlib
- Jupyter Notebook
- Joblib
- VS Code
- Git & GitHub

---

Dataset Information

The dataset contains road-related citizen complaints categorized into sentiment/severity labels.

Complaint Categories

- Critical
- Negative
- Neutral
- Positive

Dataset File

"roads.csv"

---

Project Structure

citizen-grievance-ai
│── roads.csv
│── roads_eda.ipynb
│── complaint_classifier.py
│── grievance_predictor.py
│── complaint_cleaner.py
│── grievance_loader.py
│── text_feature_builder.py
│── api.py
│── requirements.txt
│── README.md

---

Machine Learning Approach

TF-IDF Vectorization

TF-IDF (Term Frequency–Inverse Document Frequency) is used to convert complaint text into numerical values so that machine learning algorithms can process textual data effectively.

Logistic Regression

Logistic Regression was selected as the classification model because it performs efficiently in text classification problems and provides fast predictions with low computational cost.

---

Model Accuracy

The model achieved approximately:

80% Accuracy

on the Roads Grievance dataset.

---

How to Run the Project

1. Install Dependencies

pip install -r requirements.txt

2. Train the Model

Run the training file once to generate model files:

python complaint_classifier.py

This generates:

- "grievance_model.pkl"
- "text_encoder.pkl"

3. Start FastAPI Server

uvicorn api:app --reload

4. Open Swagger API Documentation

Open in browser:

http://127.0.0.1:8000/docs

You can directly test complaint predictions from the browser.

---

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

Future Scope

This grievance classification system can be extended to support additional public service departments such as:

- Roads Department
- Water Department
- Electricity Department

This would allow a unified citizen grievance system for multiple municipal services.

---

Author

Aditya Dhiraj Bhandarkar