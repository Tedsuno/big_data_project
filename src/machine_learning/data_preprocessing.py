import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower().strip()
    return text

def preprocess_data(path_to_csv, max_len=100, vocab_size=5000):
    df = pd.read_csv(path_to_csv)
    df = df[['text', 'sentiment']].dropna()
    df['text'] = df['text'].apply(clean_text)

    # 🧠 Binarisation : 1 = haine (negative), 0 = pas haine (neutral + positive)
    df = df[df['sentiment'] != 'neutral']  # on garde que negative et positive
    df['label'] = df['sentiment'].apply(lambda x: 1 if x == 'negative' else 0)

    # 🔠 Tokenisation + padding
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['text'])
    sequences = tokenizer.texts_to_sequences(df['text'])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, df['label'], test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, tokenizer
