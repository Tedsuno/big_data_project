import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower().strip()
    return text

def preprocess_data(path_to_csv, max_len=28, vocab_size=5000):
    df = pd.read_csv(path_to_csv)
    df = df[['text', 'sentiment']].dropna()
    df = df[df['sentiment'] != 'neutral']
    df['text'] = df['text'].apply(clean_text)

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['text'])

    sequences = tokenizer.texts_to_sequences(df['text'])
    X = pad_sequences(sequences, maxlen=max_len)

    Y = pd.get_dummies(df['sentiment']).values  # One-hot (shape: [n, 2])

    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    return X_train, X_test, Y_train, Y_test, tokenizer

