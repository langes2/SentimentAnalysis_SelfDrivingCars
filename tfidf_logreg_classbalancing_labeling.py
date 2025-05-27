from sklearn.utils import resample

# Separate each class
positive = labeled_data[labeled_data['Sentiment'] == 'Positive']
negative = labeled_data[labeled_data['Sentiment'] == 'Negative']
neutral = labeled_data[labeled_data['Sentiment'] == 'Neutral']

# Upsample the neutral class
neutral_upsampled = resample(neutral, 
                             replace=True,     # sample with replacement
                             n_samples=max(len(positive), len(negative)), # match majority class size
                             random_state=42)

# Combine into a new balanced dataset
balanced_data = pd.concat([positive, negative, neutral_upsampled])

# Shuffle the dataset
balanced_data = balanced_data.sample(frac=1, random_state=42)

# Split again
X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(
    balanced_data['ProcessedText'],
    balanced_data['Sentiment'],
    test_size=0.2,
    random_state=42,
    stratify=balanced_data['Sentiment']
)

# Vectorize
X_train_vec_bal = tfidf.fit_transform(X_train_bal)
X_test_vec_bal = tfidf.transform(X_test_bal)

# Train new model
clf_bal = LogisticRegression(max_iter=1000)
clf_bal.fit(X_train_vec_bal, y_train_bal)

# Predict and evaluate
y_pred_bal = clf_bal.predict(X_test_vec_bal)
report_bal = classification_report(y_test_bal, y_pred_bal, output_dict=True)
conf_matrix_bal = confusion_matrix(y_test_bal, y_pred_bal, labels=["Positive", "Neutral", "Negative"])

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix_bal, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Positive", "Neutral", "Negative"],
            yticklabels=["Positive", "Neutral", "Negative"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Balanced TF-IDF + Logistic Regression")
plt.tight_layout()
plt.show()

report_bal