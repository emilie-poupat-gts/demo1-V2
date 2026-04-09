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
# 1. Chargement du dataset
# ---------------------------------------------------------
df = pd.read_csv("./src/movies_database.csv")

X_text = df["description"]
y = df["categorie"]

# ---------------------------------------------------------
# 2. Split train/test
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
# 3. Vectorisation TF-IDF
# ---------------------------------------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------------------------------------------------
# 4. Modèle ML : Régression Logistique
# ---------------------------------------------------------
clf = LogisticRegression(
    max_iter=5000,
    solver="lbfgs",
    class_weight="balanced"
)

clf.fit(X_train_vec, y_train)

# ---------------------------------------------------------
# 5. Prédictions & Évaluation
# ---------------------------------------------------------
y_pred = clf.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print("\n🎯 Accuracy :", round(accuracy, 3))

print("\n📊 Classification Report :")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------
# 6. Matrice de confusion
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
# 7. Sauvegarde du modèle et du vectorizer
# ---------------------------------------------------------
joblib.dump(clf, "modele_ml_lr.joblib")
joblib.dump(vectorizer, "vectorizer2.joblib")

print("\Modèle et vectorizer sauvegardés.")
