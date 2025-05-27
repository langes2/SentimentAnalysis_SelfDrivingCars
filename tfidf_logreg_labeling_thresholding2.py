# Load the reuploaded dataset and run threshold-based relabeling using only the CNN human-labeled subset

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# Load dataset
df = pd.read_csv("/mnt/data/selfdriving_5class_sentiment_v2.csv")
df = df.dropna(subset=["ProcessedText", "Sentiment"])

# Extract last 299 CNN-labeled entries using the 'Sentiment' column
cnn_articles = df.tail(299)

# Train logistic regression on CNN subset
X_train = cnn_articles["ProcessedText"]
y_train = cnn_articles["Sentiment"]

# Encode labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train_vec = tfidf.fit_transform(X_train)

# Train model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train_encoded)

# Predict on the entire dataset using the trained model
X_full = df["ProcessedText"]
X_full_vec = tfidf.transform(X_full)
probs = clf.predict_proba(X_full_vec)
max_probs = probs.max(axis=1)
predicted_labels = clf.classes_[probs.argmax(axis=1)]

# Apply thresholding for 5-class relabeling
final_labels = []
for p, label_idx in zip(max_probs, predicted_labels):
    label_name = le.inverse_transform([label_idx])[0]
    if p < 0.50:
        final_labels.append("Neutral")
    elif label_name == "Positive":
        final_labels.append("Positive" if p >= 0.75 else "Slightly Positive")
    elif label_name == "Negative":
        final_labels.append("Negative" if p >= 0.78 else "Slightly Negative")
    else:
        final_labels.append("Neutral")

# Count new class distribution
cnn_thresholded_distribution = Counter(final_labels)
cnn_thresholded_distribution