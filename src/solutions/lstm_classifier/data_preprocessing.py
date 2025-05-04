import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pymongo import MongoClient
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://Ayoub:BigData123@bigdataproject.0fq9v2b.mongodb.net/")
DB_NAME = "sentiment_project"
COLLECTION_NAME = "tweets"

def preprocess_data(max_len=28, vocab_size=5000, test_size=0.33, random_state=42):
    client = None
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        cursor = collection.find({}, {'_id': 0, 'text': 1, 'sentiment': 1})
        df = pd.DataFrame(list(cursor))

        print("Preprocessing DataFrame...")
        df = df[['text', 'sentiment']].dropna()
        df = df[df['sentiment'] != 'neutral']
        print(f"Utilisation de {len(df)} documents positifs/négatifs.")

        raw_texts = df['text'].tolist()

        print("Tokenizing et padding des séquences...")
        tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
        tokenizer.fit_on_texts(raw_texts)

        sequences = tokenizer.texts_to_sequences(raw_texts)
        X = pad_sequences(sequences, maxlen=max_len)
        Y = pd.get_dummies(df['sentiment']).values
        X_train, X_test, Y_train, Y_test, texts_train, texts_test = train_test_split(
            X, Y, raw_texts, 
            test_size=test_size,
            random_state=random_state,
            stratify=Y 
        )
        return X_train, X_test, Y_train, Y_test, texts_train, texts_test, tokenizer

    except Exception as e:
        print(f"Une erreur est survenue dans preprocess_data: {e}")
        return None, None, None, None, None, None, None
    finally:
        if client:
            client.close()
            
            print("Connexion MongoDB fermée (preprocess_data).")
