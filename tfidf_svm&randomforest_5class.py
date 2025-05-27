# Load the re-uploaded file
data_5class = pd.read_csv("/mnt/data/selfdriving_5class_sentiment_v2.csv")

# Drop rows with missing text or labels
data_5class = data_5class.dropna(subset=['ProcessedText', 'Sentiment_5Class'])

# Proceed with TF-IDF vectorization and model comparisons
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

#Results: 67.2% accuracy RandomForest
#67.6% accuracy with SVM

# Split dataset
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

# Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_vec, y_train)
rf_pred = rf_clf.predict(X_test_vec)

# Evaluation
rf_report = classification_report(y_test, rf_pred, output_dict=True)

# SVM Classifier
svm_clf = LinearSVC()
svm_clf.fit(X_train_vec, y_train)
svm_pred = svm_clf.predict(X_test_vec)

# Evaluation
svm_report = classification_report(y_test, svm_pred, output_dict=True)

rf_report, svm_report