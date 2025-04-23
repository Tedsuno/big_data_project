# data_preprocessing.py
import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pymongo import MongoClient

MONGO_URI = "mongodb+srv://Ayoub:BigData123@bigdataproject.0fq9v2b.mongodb.net/"
DB_NAME = "sentiment_project"
COLLECTION_NAME = "tweets"
# -----------------------------------------
def preprocess_data(max_len=28, vocab_size=5000):
    client = None 
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        cursor = collection.find({}, {'_id': 0, 'text': 1, 'sentiment': 1})

        df = pd.DataFrame(list(cursor))
        
        print("🧹 Preprocessing DataFrame...")
        df = df[['text', 'sentiment']].dropna()
        
        df = df[df['sentiment'] != 'neutral']
        print(f"Using {len(df)} positive/negative documents for training.")

        print("Tokenizing and padding sequences...")
        tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
        tokenizer.fit_on_texts(df['text'])

        sequences = tokenizer.texts_to_sequences(df['text'])
        X = pad_sequences(sequences, maxlen=max_len)
        Y = pd.get_dummies(df['sentiment']).values
        print(f"Labels shape (Y): {Y.shape}")

        print("Splitting data into train/test sets...")
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42, stratify=Y)

        print("Preprocessing finished.")
        return X_train, X_test, Y_train, Y_test, tokenizer

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None, None
    finally:
        if client:
            client.close()
            print("MongoDB connection closed.")
