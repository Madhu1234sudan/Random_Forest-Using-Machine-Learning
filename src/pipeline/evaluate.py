from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)


    acc = accuracy_score(y_test, preds)
    print("\nAccuracy: ", acc)


    print("\nClassification Report:\n", classification_report(y_test, preds))


    print("\nConfusion Matrix:\n", confusion_matrix(y_test, preds))
    