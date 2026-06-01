import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("roads.csv")

# Features and target
X = df["complaint_text"]
y = df["sentiment"]

# Convert text to numbers
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vectorized, y)

# Take complaint from user
user_input = input("Enter complaint: ")

# Predict sentiment
user_vector = vectorizer.transform([user_input])
prediction = model.predict(user_vector)

print("Predicted Sentiment:", prediction[0])