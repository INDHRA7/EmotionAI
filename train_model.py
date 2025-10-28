import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Create folders if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Path to dataset
data_path = "data/emotion.csv"

# If no dataset exists, create a sample one
if not os.path.exists(data_path):
    sample_data = {
        "Comment": [
            "I am happy today",
            "I feel very sad",
            "This makes me angry",
            "I am excited and joyful",
            "I am scared",
            "I love this moment",
            "I am disappointed and frustrated"
        ],
        "Emotion": [
            "joy",
            "sadness",
            "anger",
            "joy",
            "fear",
            "love",
            "sadness"
        ]
    }
    pd.DataFrame(sample_data).to_csv(data_path, index=False)

# Load dataset
df = pd.read_csv(data_path)

# Basic Comment cleaning
df[""] = df["Comment"].str.lower().str.replace(f"[{string.punctuation}]", "", regex=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["Comment"], df["Emotion"], test_size=0.2, random_state=42
)

# Create ML pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('clf', MultinomialNB())
])

# Train the model
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("✅ Model training completed!")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(model, "models/Emotion_model.pkl")
print("✅ Model saved at models/Emotion_model.pkl")
