import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------
df = pd.read_csv("./src/movies_database.csv")

X_text = df["description"]
y = df["category"]

# ---------------------------------------------------------
# 2. TRAIN TEST SPLIT
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("Taille train :", len(X_train))
print("Taille test  :", len(X_test))

# ---------------------------------------------------------
# 3. TF-IDF VECTORISATION
# ---------------------------------------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------------------------------------------------
# 4. ML MODEL : LOGISTIC REGRESSION
# ---------------------------------------------------------
clf = LogisticRegression(
    max_iter=5000,
    solver="lbfgs",
    class_weight="balanced"
)

clf.fit(X_train_vec, y_train)

# ---------------------------------------------------------
# 5. ¨REDICTIONS AND EVALUATION
# ---------------------------------------------------------
y_pred = clf.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy :", round(accuracy, 3))

print("\nClassification Report :")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------
# 6. CONFUSION MATRIX
# ---------------------------------------------------------
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=clf.classes_,
    yticklabels=clf.classes_
)
plt.title("Matrice de confusion")
plt.xlabel("Prédiction")
plt.ylabel("Vérité")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 7. SAVE MODEL AND VECTORIZER
# ---------------------------------------------------------
joblib.dump(clf, "./models/modele_ml_lr.joblib")
joblib.dump(vectorizer, "./models/vectorizer2.joblib")

print("\Modèle et vectorizer saved.")
