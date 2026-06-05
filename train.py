import pandas as pd
import nltk
import joblib

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Download NLTK resources
nltk.download('stopwords')

# Load Dataset
df = pd.read_csv("all_tickets_processed_improved_v3.csv")

# Use Document column as input text
df["text"] = df["Document"].astype(str)

# Stopwords
stop_words = set(stopwords.words("english"))

# Text Cleaning Function
def clean_text(text):
    text = text.lower()

    tokens = text.split()

    tokens = [
        word for word in tokens
        if word.isalpha() and word not in stop_words
    ]

    return " ".join(tokens)

# Clean Text
df["clean_text"] = df["text"].apply(clean_text)

# Features
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(df["clean_text"])

# Target Column
print(df["Topic_group"].value_counts())

y_category = df["Topic_group"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_category,
    test_size=0.2,
    random_state=42
)

# Model
model = LinearSVC()

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nCategory Model Accuracy:", accuracy)

# Save Files
joblib.dump(model, "category_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel Saved Successfully!")