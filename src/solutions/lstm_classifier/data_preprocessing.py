# data_preprocessing.py
import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
# Importer MongoClient
from pymongo import MongoClient

MONGO_URI = "mongodb+srv://Ayoub:BigData123@bigdataproject.0fq9v2b.mongodb.net/"
DB_NAME = "sentiment_project"
COLLECTION_NAME = "tweets"
# -----------------------------------------

# Modifier la signature : plus besoin de path_to_csv
def preprocess_data(max_len=28, vocab_size=5000):
    client = None # Initialiser client à None pour le bloc finally
    try:
        print("🌍 Connecting to MongoDB Atlas...")
        # Se connecter à la base de données
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        print(f"✅ Connected to DB: '{DB_NAME}', Collection: '{COLLECTION_NAME}'")

        print("⬇️ Fetching data from MongoDB...")
        # Récupérer les données (seulement les champs 'text' et 'sentiment')
        # Le premier argument {} signifie "tous les documents"
        # Le deuxième argument spécifie les champs à inclure (1) ou exclure (0)
        # On exclut '_id' (0) et on inclut 'text' et 'sentiment' (1)
        cursor = collection.find({}, {'_id': 0, 'text': 1, 'sentiment': 1})

        # Convertir les données récupérées en DataFrame pandas
        df = pd.DataFrame(list(cursor))
        print(f"📊 Fetched {len(df)} documents into DataFrame.")

        # --- Le reste du prétraitement est identique ---
        print("🧹 Preprocessing DataFrame...")
        # Supprimer les lignes où 'text' ou 'sentiment' sont manquants (si jamais il y en a)
        df = df[['text', 'sentiment']].dropna()

        # Filtrer les sentiments 'neutral' (si la logique est toujours la même)
        # Assurez-vous que les sentiments dans MongoDB sont bien 'positive'/'negative'
        df = df[df['sentiment'] != 'neutral']
        print(f"📈 Using {len(df)} positive/negative documents for training.")

        print("🔄 Tokenizing and padding sequences...")
        tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
        tokenizer.fit_on_texts(df['text'])

        sequences = tokenizer.texts_to_sequences(df['text'])
        X = pad_sequences(sequences, maxlen=max_len)

        # Convertir les labels 'sentiment' en one-hot encoding
        # Assurez-vous que les valeurs sont bien 'positive' et 'negative'
        # S'il n'y a que 'positive'/'negative', get_dummies créera 2 colonnes
        Y = pd.get_dummies(df['sentiment']).values # One-hot (shape: [n, 2])
        # Vérifiez la forme de Y, elle doit correspondre à la sortie de votre modèle
        print(f"Labels shape (Y): {Y.shape}")


        print("✂️ Splitting data into train/test sets...")
        # Utilisation correcte de train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42, stratify=Y) # Ajout de stratify pour garder la proportion des classes

        print("✨ Preprocessing finished.")
        return X_train, X_test, Y_train, Y_test, tokenizer

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        # Retourner des valeurs vides ou None en cas d'erreur pour éviter de planter le script appelant
        return None, None, None, None, None
    finally:
        # Toujours fermer la connexion à la base de données
        if client:
            client.close()
            print("🔒 MongoDB connection closed.")