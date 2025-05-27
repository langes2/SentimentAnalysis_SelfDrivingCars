from sklearn.svm import LinearSVC

# Train SVM on class-balanced TF-IDF features
svm_clf = LinearSVC(max_iter=1000)
svm_clf.fit(X_train_vec, y_train)

# Predict and evaluate
svm_pred = svm_clf.predict(X_test_vec)
svm_accuracy = accuracy_score(y_test, svm_pred)
svm_report = classification_report(y_test, svm_pred, digits=4)

svm_accuracy, svm_report