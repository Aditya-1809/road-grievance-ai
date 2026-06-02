
# Citizen Grievance AI - Roads Department

## Project Overview
Citizen Grievance AI is an NLP-based complaint classification system developed for the Roads Department. The system analyzes citizen complaints and predicts the sentiment category of road-related issues.

The model classifies complaints into:

- Critical
- Negative
- Neutral
- Positive

Example:

Input:
```text
Huge potholes near station causing accidents
```

Output:
```text
Critical
```

---

## Features
- NLP-based complaint sentiment analysis
- TF-IDF text vectorization
- Logistic Regression model
- Complaint prediction system
- Exploratory Data Analysis (EDA)
- Confusion Matrix visualization
- Word Cloud generation

---

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook
- VS Code

---

## Dataset
The dataset contains road-related citizen complaints categorized into sentiment labels:

- Critical
- Negative
- Neutral
- Positive

Dataset File:
`roads.csv`

---

## Project Structure

```text
citizen-grievance-ai
│── roads.csv
│── roads_eda.ipynb
│── train_model.py
│── predict.py
│── clean_text.py
│── README.md
```

---

## Model Used

### TF-IDF Vectorization
Used to convert complaint text into numerical form for machine learning.

### Logistic Regression
Used as the classification algorithm for sentiment prediction.

---

## Model Accuracy
Achieved approximately:

**80%**

accuracy on the Roads grievance dataset.

---

## How to Run

### Install dependencies
```bash
pip install pandas scikit-learn matplotlib jupyter notebook
```

### Train model
```bash
python train_model.py
```

### Predict complaint sentiment
```bash
python predict.py
```

---

## Sample Prediction

Input:
```text
Road repaired successfully
```

Output:
```text
Positive
```

---

## Author
Aditya Bhandarakar