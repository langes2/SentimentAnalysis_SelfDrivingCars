import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

#Results: 86.4% accuracy

# Load the dataset
df = pd.read_csv("/mnt/data/selfdriving_5class_sentiment_v2.csv")

# Drop missing values and prepare for balancing
df = df.dropna(subset=['ProcessedText', 'Sentiment_5Class'])

# Separate by class and apply random oversampling
classes = df['Sentiment_5Class'].unique()
dfs = [df[df['Sentiment_5Class'] == c] for c in classes]
max_size = max(len(d) for d in dfs)
resampled = [resample(d, replace=True, n_samples=max_size, random_state=42) for d in dfs]
df_balanced = pd.concat(resampled).sample(frac=1, random_state=42)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df_balanced['ProcessedText'], df_balanced['Sentiment_5Class'],
    test_size=0.2, random_state=42, stratify=df_balanced['Sentiment_5Class']
)

# Vectorize with TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# Train Logistic Regression
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)

# Evaluate
y_pred = clf.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=4)

accuracy, report