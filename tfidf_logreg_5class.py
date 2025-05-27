from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Results: 67.2% accuracy
# Biased towards dominant labels

# Reload the 5-class dataset
data_5class = pd.read_csv("/mnt/data/selfdriving_5class_sentiment_v2.csv")

# Drop rows with missing labels or text
data_5class = data_5class.dropna(subset=['ProcessedText', 'Sentiment_5Class'])

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    data_5class['ProcessedText'],
    data_5class['Sentiment_5Class'],
    test_size=0.2,
    random_state=42,
    stratify=data_5class['Sentiment_5Class']
)

# TF-IDF vectorization
tfidf_vec = TfidfVectorizer(max_features=5000)
X_train_vec = tfidf_vec.fit_transform(X_train)
X_test_vec = tfidf_vec.transform(X_test)

# Train logistic regression
clf_5class = LogisticRegression(max_iter=1000)
clf_5class.fit(X_train_vec, y_train)

# Predictions and evaluation
y_pred = clf_5class.predict(X_test_vec)
report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred, labels=clf_5class.classes_)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=clf_5class.classes_,
            yticklabels=clf_5class.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - TF-IDF + Logistic Regression (5-class)")
plt.tight_layout()
plt.show()

report