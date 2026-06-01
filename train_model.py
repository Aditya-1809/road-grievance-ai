import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("roads.csv")

# Features and target
X = df["complaint_text"]
y = df["sentiment"]

# Convert text to numbers
vectorizer = TfidfVectorizer(
    ngram_range=(1,3),
    stop_words="english",
    max_features=3000
)

X_vectorized = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.1,
    random_state=42
)

# Train model
model = LogisticRegression(
    max_iter=3000,
    C=2,
    solver="lbfgs"
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy * 100, "%")