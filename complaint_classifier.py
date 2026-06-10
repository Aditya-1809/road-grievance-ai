import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
road_complaints = pd.read_csv("roads.csv")

# Remove empty and duplicate rows
road_complaints = road_complaints.dropna()
road_complaints = road_complaints.drop_duplicates()

# Features and target
complaint_messages = road_complaints["complaint_text"]
complaint_sentiment = road_complaints["sentiment"]

# Better text processing
text_encoder = TfidfVectorizer(
    ngram_range=(1, 4),
    stop_words="english",
    max_features=6000,
    min_df=1,
    sublinear_tf=True
)

# Convert text into vectors
processed_features = text_encoder.fit_transform(
    complaint_messages
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    processed_features,
    complaint_sentiment,
    test_size=0.15,
    random_state=42,
    stratify=complaint_sentiment
)

# Improved model
grievance_classifier = LogisticRegression(
    max_iter=5000,
    C=15,
    solver="liblinear",
    class_weight="balanced",
    random_state=42
)

# Train model
grievance_classifier.fit(X_train, y_train)

# Predict
predicted_output = grievance_classifier.predict(
    X_test
)

# Accuracy
accuracy = accuracy_score(
    y_test,
    predicted_output
)

# Save model and vectorizer
joblib.dump(
    grievance_classifier,
    "grievance_model.pkl"
)

joblib.dump(
    text_encoder,
    "text_encoder.pkl"
)

print("Model Accuracy:", round(accuracy * 100, 2), "%")
print("Model saved successfully.")
print("Vectorizer saved successfully.")