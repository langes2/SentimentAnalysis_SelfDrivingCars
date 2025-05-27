import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

#Results: 94% accuracy!

# 1. Load dataset
df = pd.read_csv("selfdriving_5class_sentiment_v2.csv")
df = df.dropna(subset=["ProcessedText", "Sentiment_5Class"])

# 2. Random oversampling (class balancing)
classes = df['Sentiment_5Class'].unique()
dfs = [df[df['Sentiment_5Class'] == c] for c in classes]
max_size = max(len(d) for d in dfs)
resampled = [resample(d, replace=True, n_samples=max_size, random_state=42) for d in dfs]
df_balanced = pd.concat(resampled).sample(frac=1, random_state=42).reset_index(drop=True)

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df_balanced['ProcessedText'], df_balanced['Sentiment_5Class'],
    test_size=0.2, random_state=42, stratify=df_balanced['Sentiment_5Class']
)

# 4. TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# 5. Train Random Forest classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_vec, y_train)

# 6. Predict and evaluate
y_pred = rf_clf.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=4)

# 7. Print results
print(f"\nTF-IDF + Random Forest (Balanced) Accuracy: {accuracy:.4f}\n")
print(report)