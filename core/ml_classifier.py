def classify(description, clf, vectorizer):
    vec = vectorizer.transform([description])
    probas = clf.predict_proba(vec)[0]

    predicted = clf.classes_[probas.argmax()]
    confidence = probas.max()

    return predicted, float(confidence)
