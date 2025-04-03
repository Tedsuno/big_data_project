import pandas as pd
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay # Ajout
import tensorflow as tf # Bonne pratique d'importer tf directement
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional # Ajout Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping # Ajout EarlyStopping

#nltk.download('punkt') # Déjà fait normalement

# === Nettoyage des tweets ===
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text) # Supprime les mentions
    # text = re.sub(r'#', '', text) # Option 1: Supprime juste #, garde le mot
    text = re.sub(r'[^\w\s\#]', '', text) # Garde les hashtags comme partie du mot (ou supprimez '#' ici si Option 1 choisie)
    # Garder les emojis ou les convertir en texte serait mieux ici
    text = re.sub(r'\d+', '', text)
    # Optionnel: Lemmatisation/Stemming ici
    return text.strip()

# === Transformation du sentiment en label binaire ===
# Assurez-vous que 0 = classe minoritaire (hate) et 1 = classe majoritaire (non-hate)
# Si c'est l'inverse, ajustez les poids et l'analyse d'erreur
def sentiment_to_label(sentiment):
    # Adaptez ceci exactement à vos labels CSV (ex: 'hate'/'non-hate' ou 'negative'/'positive')
    return 0 if str(sentiment).lower() == "negative" else 1 # ou "hate"

# === Chargement et prétraitement du CSV ===
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # Vérifiez les noms de colonnes réels dans votre CSV
    df['clean_text'] = df['text'].astype(str).apply(clean_text)
    df['label'] = df['sentiment'].apply(sentiment_to_label)
    # Supprimer les textes vides après nettoyage
    df = df[df['clean_text'].str.len() > 0]
    return df

# === Tokenization & padding ===
# Déterminez max_len basé sur l'analyse des longueurs de séquences (ex: 95 percentile)
def prepare_sequences(texts, max_words=10000, max_len=120): # Augmenter potentiellement max_words, ajuster max_len
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post') # Ajouter truncating
    return padded, tokenizer, max_len # Retourner max_len utilisé

# === Modèle LSTM ===
def build_lstm_model(vocab_size, embedding_dim, input_length): # Ajouter embedding_dim comme paramètre
    model = Sequential()
    # Option: Ajouter ici le chargement de poids pré-entraînés si utilisés
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length))
    # Amélioration: Utiliser Bidirectional LSTM
    model.add(Bidirectional(LSTM(128, return_sequences=False))) # Ou 64, 128 - à tester. return_sequences=True si on empile
    # model.add(LSTM(64)) # Si on empile
    model.add(Dropout(0.4)) # Taux de dropout à tester (0.2 - 0.5)
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # Adam avec learning rate ajustable
        metrics=['accuracy', Precision(name='precision'), Recall(name='recall')] # Donner des noms explicites aux métriques
    )
    return model

# === Pipeline complet ===
def train_model(csv_path):
    df = load_data(csv_path)

    print("📊 Répartition des classes après chargement :")
    print(df['label'].value_counts())

    # Analyser la longueur des textes nettoyés pour choisir max_len
    df['text_len'] = df['clean_text'].apply(lambda x: len(x.split()))
    max_len_calculated = int(df['text_len'].quantile(0.95)) # Ex: 95ème percentile
    print(f"📊 Longueur de séquence choisie (95 percentile) : {max_len_calculated}")

    # Paramètres
    MAX_VOCAB_SIZE = 10000 # Hyperparamètre
    EMBEDDING_DIM = 100 # Hyperparamètre (souvent 100, 200, 300 pour pré-entraînés)
    MAX_LEN = max_len_calculated # Utiliser la longueur calculée
    EPOCHS = 20 # Augmenter le nombre d'époques
    BATCH_SIZE = 64 # Hyperparamètre
    VALIDATION_SPLIT = 0.15 # Fraction pour le set de validation

    X, tokenizer, actual_max_len = prepare_sequences(df['clean_text'], max_words=MAX_VOCAB_SIZE, max_len=MAX_LEN)
    y = df['label'].values

    # Division Train / Validation / Test
    # D'abord, séparer le Test set
    X_temp, X_test, y_temp, y_test, df_temp, df_test_final = train_test_split(
        X, y, df, test_size=0.2, random_state=42, stratify=y # Stratify pour garder la proportion des classes
    )
    # Ensuite, séparer Train / Validation depuis X_temp
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VALIDATION_SPLIT / (1 - 0.2), random_state=42, stratify=y_temp # Ajuster la proportion pour le split
    )

    print(f"Taille Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

    # === Calculer les poids des classes (sur le train set) ===
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    print("\n⚖️ Poids des classes calculés (balanced) :", class_weights_dict)

    # === Callback Early Stopping ===
    early_stopping = EarlyStopping(
        monitor='val_loss', # Surveiller la perte sur le set de validation
        patience=3,          # Nombre d'époques sans amélioration avant d'arrêter
        restore_best_weights=True # Garder les meilleurs poids trouvés
    )

    # === Entraînement ===
    model = build_lstm_model(vocab_size=MAX_VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, input_length=actual_max_len)
    print(model.summary()) # Afficher la structure du modèle

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val), # Utiliser le VRAI set de validation
        class_weight=class_weights_dict,
        callbacks=[early_stopping] # Ajouter le callback
    )

    # === Évaluation sur le Test Set ===
    print("\nÉvaluation sur le Test Set:")
    loss, acc, precision, recall = model.evaluate(X_test, y_test, verbose=1)
    print(f"✅ Accuracy : {acc*100:.2f}%")
    print(f"📐 Precision : {precision:.2f}")
    print(f"🎯 Recall : {recall:.2f}")
    # Calcul du F1 Score manuel si non inclus dans les métriques Keras standards
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"📊 F1 Score : {f1_score:.2f}")


    # === Prédictions & Analyse ===
    y_pred_proba = model.predict(X_test) # Probabilités

    # Trouver le meilleur seuil sur le set de validation (exemple simple, on pourrait optimiser F1)
    # Note: Ceci est une simplification. Une recherche de seuil plus rigoureuse serait mieux.
    # Pour cet exemple, on garde 0.5 ou on utilise celui que vous aviez avant si justifié.
    threshold = 0.5 # Utiliser 0.5 comme défaut, ajuster si nécessaire basé sur val set analysis

    print(f"\nSeuil de décision utilisé : {threshold}")
    y_pred_labels = (y_pred_proba >= threshold).astype(int).flatten() # Classes prédites (0 ou 1)

    # Rapport de classification détaillé
    print("\nClassification Report (Test Set):")
    # Assurez-vous que target_names correspond bien à 0 et 1
    print(classification_report(y_test, y_pred_labels, target_names=["Hate (0)", "Non-Hate (1)"]))

    # Matrice de Confusion
    print("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Hate (0)", "Non-Hate (1)"])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()


    # === Histogramme des scores
    plt.hist(y_pred_proba, bins=50)
    plt.title("Distribution des probabilités prédites (Test Set)")
    plt.xlabel("Proba prédite (classe 1 - Non-Hate)")
    plt.ylabel("Nombre de tweets")
    plt.axvline(threshold, color='r', linestyle='dashed', linewidth=1, label=f'Seuil={threshold}')
    plt.legend()
    plt.show()

    # === Sauvegarde dans CSV avec infos utiles
    df_test_final = df_test_final.copy()
    df_test_final['score_proba_non_hate'] = y_pred_proba
    df_test_final['predicted_label'] = y_pred_labels
    df_test_final['prediction'] = df_test_final['predicted_label'].map({0: "hate", 1: "non-hate"})
    df_test_final['correct_prediction'] = df_test_final['label'] == df_test_final['predicted_label']

    df_test_final.to_csv("predictions_detailed.csv", index=False)
    print("\n📁 Fichier 'predictions_detailed.csv' généré.")

    # === Afficher les erreurs spécifiques ===
    false_negatives = df_test_final[(df_test_final['label'] == 0) & (df_test_final['predicted_label'] == 1)]
    false_positives = df_test_final[(df_test_final['label'] == 1) & (df_test_final['predicted_label'] == 0)]

    print(f"\n⚠️ Faux Négatifs (Hate non détecté) : {len(false_negatives)}")
    print(false_negatives[['text', 'clean_text', 'score_proba_non_hate', 'prediction']].head(10))

    print(f"\n⚠️ Faux Positifs (Non-Hate classé comme Hate) : {len(false_positives)}")
    print(false_positives[['text', 'clean_text', 'score_proba_non_hate', 'prediction']].head(10))

    return model, tokenizer, history # Retourner aussi l'historique pour analyse

# === Lancer le script ===
if __name__ == "__main__":
    # Assurez-vous que le chemin est correct ou passez-le en argument
    csv_file_path = "C:/Users/amine/big_data_project/data/tweet.csv"
    model, tokenizer, history = train_model(csv_file_path)

    # Optionnel: Tracer les courbes d'apprentissage
    if history:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss Evolution')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy Evolution')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()