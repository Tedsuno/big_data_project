import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import numpy as np

# 1. Charger le fichier CSV
df = pd.read_csv('../../data/tweet.csv')


# Supprimer les lignes où le texte est manquant
df = df.dropna(subset=['text', 'sentiment'])

# (Optionnel) Réinitialiser les index si tu veux
df = df.reset_index(drop=True)

# 2. Utiliser les colonnes correctes
X = df['text']           # Texte du tweet
y = df['sentiment']      # Label (positive, negative, neutral)

# 3. Séparer en train / test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Entraîner le modèle SVM
clf = LinearSVC()
clf.fit(X_train_vec, y_train)

# 6. Prédictions
y_pred = clf.predict(X_test_vec)

# 7. Rapport d’évaluation
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np

# 1. Matrice de confusion
labels = sorted(df['sentiment'].unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Matrice de Confusion")
plt.xlabel("Prédiction")
plt.ylabel("Vrai label")
plt.tight_layout()
plt.show()

# 2. Barplot des scores par classe
report = classification_report(y_test, y_pred, output_dict=True)
f1_scores = {label: report[label]['f1-score'] for label in labels}

plt.figure(figsize=(8,6))
sns.barplot(x=list(f1_scores.keys()), y=list(f1_scores.values()))
plt.ylim(0, 1)
plt.title("F1-Score par classe")
plt.xlabel("Classe")
plt.ylabel("F1-Score")
plt.tight_layout()
plt.show()

# Pour enregistrer les performances
train_sizes = [0.1, 0.3, 0.5, 0.7, 0.99]
f1_scores = []

for size in train_sizes:
    # Prendre une fraction du dataset
    X_partial, _, y_partial, _ = train_test_split(X_train, y_train, train_size=size, random_state=42)
    
    # Vectorisation
    X_partial_vec = vectorizer.fit_transform(X_partial)
    X_test_vec = vectorizer.transform(X_test)
    
    # Modèle
    model = LinearSVC()
    model.fit(X_partial_vec, y_partial)
    
    # Prédictions
    y_partial_pred = model.predict(X_test_vec)
    report_partial = classification_report(y_test, y_partial_pred, output_dict=True)
    
    # Moyenne des F1-scores (macro pour donner le même poids à chaque classe)
    f1_macro = report_partial['macro avg']['f1-score']
    f1_scores.append(f1_macro)

# Affichage
plt.figure(figsize=(8,6))
plt.plot([int(x*100) for x in train_sizes], f1_scores, marker='o')
plt.title("Évolution du F1-Score selon la taille du jeu d'entraînement")
plt.xlabel("Pourcentage de données d'entraînement (%)")
plt.ylabel("F1-score (macro)")
plt.grid(True)