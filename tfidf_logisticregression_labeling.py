from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Extract only labeled data for training
labeled_data = merged_data.dropna(subset=['Sentiment'])

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    labeled_data['ProcessedText'],
    labeled_data['Sentiment'],
    test_size=0.2,
    random_state=42,
    stratify=labeled_data['Sentiment']
)

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# Train logistic regression model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)

# Predictions and evaluation
y_pred = clf.predict(X_test_vec)
report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred, labels=["Positive", "Neutral", "Negative"])

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Positive", "Neutral", "Negative"],
            yticklabels=["Positive", "Neutral", "Negative"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - TF-IDF + Logistic Regression")
plt.tight_layout()
plt.show()

report