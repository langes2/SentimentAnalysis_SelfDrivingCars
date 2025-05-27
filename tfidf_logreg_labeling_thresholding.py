# Retry the two-layer thresholding on the re-uploaded dataset

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# Reload dataset
df = pd.read_csv("/mnt/data/selfdriving_5class_sentiment_v2.csv")
df = df.dropna(subset=["ProcessedText", "Sentiment_5Class"])
df = df[df["Sentiment_5Class"] != "Neutral"]

# Encode labels
le = LabelEncoder()
df["label"] = le.fit_transform(df["Sentiment_5Class"])

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_vec = tfidf.fit_transform(df["ProcessedText"])

# Train logistic regression
clf = LogisticRegression(max_iter=1000)
clf.fit(X_vec, df["label"])

# Predict probabilities
probs = clf.predict_proba(X_vec)
max_probs = probs.max(axis=1)
predicted_labels = clf.classes_[probs.argmax(axis=1)]

# Two-layer thresholding
threshold_neutral = 0.45
threshold_strong_sentiment = 0.80

final_labels = []
for p, label_idx in zip(max_probs, predicted_labels):
    label_name = le.inverse_transform([label_idx])[0]
    if p < threshold_neutral:
        final_labels.append("Neutral")
    elif label_name == "Slightly Positive" and p > threshold_strong_sentiment:
        final_labels.append("Positive")
    elif label_name == "Slightly Negative" and p > threshold_strong_sentiment:
        final_labels.append("Negative")
    else:
        final_labels.append(label_name)

# Calculate new class distribution
updated_class_distribution = Counter(final_labels)
updated_class_distribution