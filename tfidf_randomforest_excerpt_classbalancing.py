from sklearn.ensemble import RandomForestClassifier

# Train Random Forest on class-balanced TF-IDF features
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_vec, y_train)

# Predict and evaluate
rf_pred = rf_clf.predict(X_test_vec)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_report = classification_report(y_test, rf_pred, digits=4)

rf_accuracy, rf_report