# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Analyse Approfondie et Visualisations par Modèle
#
# Objectifs :
# * Charger les données depuis MongoDB.
# * Préparer les données et obtenir les prédictions pour chaque modèle (SVM, Lexique, LSTM).
# * Générer des visualisations spécifiques pour comprendre le comportement de chaque modèle :
#     * **Lexique (TextBlob) :** Analyse de Polarité/Subjectivité.
#     * **SVM (TF-IDF) :** Importance des N-grammes (features).
#     * **LSTM :** Word Clouds des exemples bien/mal classifiés.

# %%
# Imports généraux
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import re
import os
from collections import Counter

# Juste après les imports généraux
try:
    from IPython.display import display
except ImportError:
    display = print

# MongoDB
from pymongo import MongoClient
from dotenv import load_dotenv

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# TextBlob
from textblob import TextBlob

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Word Cloud
from wordcloud import WordCloud, STOPWORDS

# Configuration
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 7)
# Directory to save plots
PLOTS_DIR = "plots/model_analysis"
os.makedirs(PLOTS_DIR, exist_ok=True)

# %% [markdown]
# ## 1. Chargement et Préparation des Données depuis MongoDB

# %%
# --- MongoDB Configuration ---
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://Ayoub:BigData123@bigdataproject.0fq9v2b.mongodb.net/")
DB_NAME = "sentiment_project"
COLLECTION_NAME = "tweets" # Assumes original data collection
# ---------------------

def load_data_from_mongo():
    client = None
    df = None
    try:
        print("🌍 Connecting to MongoDB Atlas...")
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        print(f"✅ Connected to DB: '{DB_NAME}', Collection: '{COLLECTION_NAME}'")

        print("⬇️ Fetching data (text, sentiment)...")
        # Fetch only needed fields
        cursor = collection.find({}, {'_id': 0, 'text': 1, 'sentiment': 1})
        df = pd.DataFrame(list(cursor))
        print(f"📊 Fetched {len(df)} documents.")

        # Basic Cleaning
        df.dropna(subset=['text', 'sentiment'], inplace=True)
        valid_sentiments = ['positive', 'negative', 'neutral']
        df = df[df['sentiment'].isin(valid_sentiments)]
        df = df.reset_index(drop=True)
        print(f"Data shape after cleaning: {df.shape}")

    except Exception as e:
        print(f"❌ An error occurred fetching data: {e}")
    finally:
        if client:
            client.close()
            print("🔒 MongoDB connection closed.")
    return df

df = load_data_from_mongo()

# Split data (using the same random_state as before)
if df is not None:
    X = df['text']
    y = df['sentiment']
    labels = sorted(y.unique())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nData split: Train={len(X_train)}, Test={len(X_test)}")
else:
    print("\nSkipping further analysis due to data loading issues.")


# %% [markdown]
# ## 2. Analyse Spécifique : Lexique (TextBlob)
#
# On analyse les scores de polarité et subjectivité donnés par TextBlob.

# %%
if df is not None:
    print("\n--- Analyse TextBlob (Lexique) ---")

    # Apply TextBlob to the test set texts
    print("Calculating Polarity and Subjectivity for Test Set...")
    textblob_sentiments = []
    polarities = []
    subjectivities = []

    # Reuse functions from previous notebook/script
    def preprocess_text_lex(text):
        if not isinstance(text, str): return ""
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^\w\s]', '', text) # Simpler cleaning for TextBlob
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.lower()
        return text

    def get_polarity_subjectivity(text):
      if not isinstance(text, str) or not text:
        return 0.0, 0.0 # Neutral default
      try:
          analysis = TextBlob(text)
          return analysis.sentiment.polarity, analysis.sentiment.subjectivity
      except Exception:
           return 0.0, 0.0

    X_test_cleaned_lex = X_test.apply(preprocess_text_lex)
    for text in X_test_cleaned_lex:
        polarity, subjectivity = get_polarity_subjectivity(text)
        polarities.append(polarity)
        subjectivities.append(subjectivity)

    # Create a DataFrame for analysis
    df_test_blob = pd.DataFrame({
        'text': X_test,
        'true_sentiment': y_test,
        'polarity': polarities,
        'subjectivity': subjectivities
    })

    # Assign predicted sentiment based on polarity (using the same threshold as before)
    def get_sentiment_pred(polarity, threshold=0.1):
        if polarity > threshold: return "positive"
        elif polarity < -threshold: return "negative"
        else: return "neutral"

    df_test_blob['predicted_sentiment_lex'] = df_test_blob['polarity'].apply(get_sentiment_pred)

    print("Calculations complete.")
    display(df_test_blob.head())

    # %% [markdown]
    # ### Visualisation Polarité vs Subjectivité

    # %%
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df_test_blob, x='polarity', y='subjectivity', hue='true_sentiment', palette='viridis', alpha=0.6, s=50)
    plt.title('Polarité vs Subjectivité des Tweets (Coloré par Vrai Sentiment)')
    plt.xlabel('Polarité (-1 à +1)')
    plt.ylabel('Subjectivité (0 à 1)')
    plt.grid(True)
    plt.axvline(0, color='grey', linestyle='--', lw=1) # Ligne pour polarité neutre
    plt.legend(title='Vrai Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, 'lexicon_polarity_vs_subjectivity.png')
    plt.savefig(plot_path)
    print(f"Saved plot: {plot_path}")
    plt.show()

    # %% [markdown]
    # ### Distribution de la Polarité par Vrai Sentiment

    # %%
    plt.figure(figsize=(12, 6))
    for sentiment in labels:
        sns.histplot(df_test_blob[df_test_blob['true_sentiment'] == sentiment]['polarity'],
                     kde=True, label=sentiment, bins=30) # bins=30 for more detail
    plt.title('Distribution de la Polarité (TextBlob) par Vrai Sentiment')
    plt.xlabel('Score de Polarité')
    plt.ylabel('Fréquence')
    plt.legend(title='Vrai Sentiment')
    plt.grid(True)
    plot_path = os.path.join(PLOTS_DIR, 'lexicon_polarity_distribution.png')
    plt.savefig(plot_path)
    print(f"Saved plot: {plot_path}")
    plt.show()

# %% [markdown]
# ## 3. Analyse Spécifique : SVM (TF-IDF)
#
# On identifie les mots (N-grammes) les plus importants pour chaque classe selon le modèle SVM.

# %%
if df is not None:
    print("\n--- Analyse SVM (TF-IDF Features) ---")

    # Re-train SVM briefly to get coefficients (or load saved model/vectorizer)
    print("Préparation TF-IDF et re-entraînement SVM rapide...")
    vectorizer_svm = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2)) # Use unigrams & bigrams
    X_train_vec_svm = vectorizer_svm.fit_transform(X_train)
    X_test_vec_svm = vectorizer_svm.transform(X_test)

    svm_model_analysis = LinearSVC(dual="auto", random_state=42, C=0.5)
    svm_model_analysis.fit(X_train_vec_svm, y_train)
    svm_pred_analysis = svm_model_analysis.predict(X_test_vec_svm)
    print("Prédictions SVM pour analyse terminées.")
    print(f"SVM Accuracy (vérification): {accuracy_score(y_test, svm_pred_analysis):.4f}")


    # %% [markdown]
    # ### Top N-grammes par Classe (Importance des Features SVM)
    #
    # On regarde les coefficients du modèle LinearSVC pour voir quels mots/bigrammes ont le plus d'influence positive pour chaque classe.

    # %%
    feature_names = np.array(vectorizer_svm.get_feature_names_out())
    n_top_features = 20

    plt.figure(figsize=(15, len(labels) * 5)) # Ajuster la taille
    plot_index = 1

    # Si le modèle est OvR (One-vs-Rest), il y a un coefficient par classe
    if len(svm_model_analysis.classes_) == len(svm_model_analysis.coef_):
        for i, label in enumerate(svm_model_analysis.classes_):
            try:
                # Obtenir les coefficients pour cette classe
                coef = svm_model_analysis.coef_[i]

                # Indices des features triés par coefficient (décroissant)
                top_positive_indices = np.argsort(coef)[-n_top_features:]
                top_positive_features = feature_names[top_positive_indices]
                top_positive_coefs = coef[top_positive_indices]

                # Indices des features triés par coefficient (croissant -> les plus négatifs)
                # Ces features sont "contre" cette classe
                # top_negative_indices = np.argsort(coef)[:n_top_features]
                # top_negative_features = feature_names[top_negative_indices]
                # top_negative_coefs = coef[top_negative_indices]

                # Plot pour les features les plus positives pour cette classe
                plt.subplot(len(labels), 1, plot_index)
                colors = ['green' if c > 0 else 'red' for c in top_positive_coefs] # Color code might be simple
                plt.barh(np.arange(n_top_features), top_positive_coefs, color=colors, align='center')
                plt.yticks(np.arange(n_top_features), top_positive_features)
                plt.xlabel("Coefficient Weight")
                plt.title(f"Top {n_top_features} N-grammes les plus influents pour la classe '{label}' (SVM)")
                plt.gca().invert_yaxis() # Afficher le plus important en haut
                plot_index += 1

            except IndexError:
                 print(f"Skipping feature importance plot for class {label} due to coefficient mismatch.")

        plt.tight_layout()
        plot_path = os.path.join(PLOTS_DIR, 'svm_top_features_per_class.png')
        plt.savefig(plot_path)
        print(f"\nSaved plot: {plot_path}")
        plt.show()
    else:
        print("Impossible d'afficher l'importance par classe (format de coefficients non attendu).")


# %% [markdown]
# ## 4. Analyse Spécifique : LSTM
#
# On visualise les mots fréquents dans les tweets que l'LSTM a bien ou mal classifiés.

# %%
if df is not None:
    print("\n--- Analyse LSTM (erreurs de classification) ---")

    # --- Re-run LSTM prediction (or load saved predictions/model) ---
    # Utiliser le modèle 3 classes du notebook de comparaison pour cohérence
    print("Préparation des données pour LSTM (3 classes)...")
    VOCAB_SIZE_LSTM = 5000
    MAX_LEN_LSTM = 30
    EMBED_DIM_LSTM = 64
    LSTM_OUT_LSTM = 100
    EPOCHS_LSTM = 4 # Re-entraîner rapidement
    BATCH_SIZE_LSTM = 64

    def clean_text_lstm(text):
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'@\w+|#', '', text)
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    X_train_clean_lstm = X_train.apply(clean_text_lstm)
    X_test_clean_lstm = X_test.apply(clean_text_lstm)

    tokenizer_lstm = Tokenizer(num_words=VOCAB_SIZE_LSTM, oov_token="<OOV>")
    tokenizer_lstm.fit_on_texts(X_train_clean_lstm)
    X_train_pad_lstm = pad_sequences(tokenizer_lstm.texts_to_sequences(X_train_clean_lstm), maxlen=MAX_LEN_LSTM, padding='post', truncating='post')
    X_test_pad_lstm = pad_sequences(tokenizer_lstm.texts_to_sequences(X_test_clean_lstm), maxlen=MAX_LEN_LSTM, padding='post', truncating='post')

    label_to_index_lstm = {label: i for i, label in enumerate(labels)}
    index_to_label_lstm = {i: label for label, i in label_to_index_lstm.items()}
    y_train_cat_lstm = to_categorical(y_train.map(label_to_index_lstm), num_classes=len(labels))
    # y_test_cat_lstm = to_categorical(y_test.map(label_to_index_lstm), num_classes=len(labels)) # Pas besoin si on compare aux labels textes

    print("Re-entraînement rapide LSTM (3 classes)...")
    model_lstm_analysis = Sequential(name="LSTM_3_Classes_Analysis")
    model_lstm_analysis.add(Embedding(input_dim=VOCAB_SIZE_LSTM, output_dim=EMBED_DIM_LSTM, input_length=MAX_LEN_LSTM))
    model_lstm_analysis.add(SpatialDropout1D(0.4))
    model_lstm_analysis.add(LSTM(LSTM_OUT_LSTM, dropout=0.2, recurrent_dropout=0.2))
    model_lstm_analysis.add(Dense(len(labels), activation='softmax'))
    model_lstm_analysis.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_lstm_analysis.fit(X_train_pad_lstm, y_train_cat_lstm, epochs=EPOCHS_LSTM, batch_size=BATCH_SIZE_LSTM, verbose=0) # verbose=0 pour moins d'output

    print("Prédictions LSTM pour analyse...")
    lstm_pred_probs_analysis = model_lstm_analysis.predict(X_test_pad_lstm)
    lstm_pred_indices_analysis = np.argmax(lstm_pred_probs_analysis, axis=1)
    lstm_pred_analysis = [index_to_label_lstm[idx] for idx in lstm_pred_indices_analysis]

    # Create DataFrame with predictions
    df_test_lstm = pd.DataFrame({
        'text': X_test,
        'cleaned_text': X_test_clean_lstm, # Use cleaned text for word cloud
        'true_sentiment': y_test,
        'predicted_sentiment_lstm': lstm_pred_analysis
    })
    df_test_lstm['correctly_classified'] = df_test_lstm['true_sentiment'] == df_test_lstm['predicted_sentiment_lstm']
    print("Prédictions LSTM pour analyse terminées.")
    display(df_test_lstm.head())

    # %% [markdown]
    # ### Word Clouds pour les Erreurs de Classification LSTM

    # %%
    stopwords_wc = set(STOPWORDS)
    stopwords_wc.update(["amp", "im", "go", "get", "dont", "u", "cant", "day", "like", "im", "lol", "ok"]) # Add more common noise words

    def plot_word_cloud(text_series, title, filename):
        if text_series.empty:
            print(f"Skipping '{title}': No text data.")
            return
        text = " ".join(review for review in text_series if isinstance(review, str))
        if not text:
             print(f"Skipping '{title}': No text content after join.")
             return

        wordcloud = WordCloud(stopwords=stopwords_wc, background_color="white", max_words=100, width=800, height=400).generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(title, fontsize=16)
        plot_path = os.path.join(PLOTS_DIR, filename)
        plt.savefig(plot_path)
        print(f"Saved plot: {plot_path}")
        plt.show()

    # Word Cloud pour les tweets mal classifiés
    misclassified_text = df_test_lstm[~df_test_lstm['correctly_classified']]['cleaned_text']
    plot_word_cloud(misclassified_text, "Mots fréquents dans les Tweets Mal Classifiés (LSTM)", "lstm_misclassified_wordcloud.png")

    # Word Cloud pour les tweets bien classifiés
    correctly_classified_text = df_test_lstm[df_test_lstm['correctly_classified']]['cleaned_text']
    plot_word_cloud(correctly_classified_text, "Mots fréquents dans les Tweets Bien Classifiés (LSTM)", "lstm_correctly_classified_wordcloud.png")

    # Optionnel : Word Clouds par classe mal classifiée
    # Exemple : Tweets positifs classifiés comme négatifs
    # misclassified_pos_as_neg = df_test_lstm[
    #    (df_test_lstm['true_sentiment'] == 'positive') &
    #    (df_test_lstm['predicted_sentiment_lstm'] == 'negative')
    # ]['cleaned_text']
    # plot_word_cloud(misclassified_pos_as_neg, "Positive Tweets Misclassified as Negative (LSTM)", "lstm_misclassified_pos_as_neg.png")


# %% [markdown]
# ## Conclusion de l'Analyse par Modèle
#
# [**TODO :** Interpréter les graphiques ici.]
# * **TextBlob :** Comment les scores de polarité/subjectivité se rapportent-ils aux vrais sentiments ? Y a-t-il des chevauchements ? La subjectivité aide