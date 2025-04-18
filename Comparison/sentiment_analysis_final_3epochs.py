
# === Imports ===
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# === Chargement et préparation des données ===
print("\n🔎 Chargement des données...")
df = pd.read_csv("C:/Users/dadzo/bigdata/big_data_project/data/tweet.csv", encoding='ISO-8859-1')
df = df.dropna(subset=["text", "sentiment"])
df = df[df["sentiment"].isin(["positive", "neutral", "negative"])]

X_train, X_test, y_train, y_test = train_test_split(df["text"], df["sentiment"], test_size=0.2, random_state=42)

# ==============================================================================
# === MODELE 1 : SVM ===========================================================
# ==============================================================================
print("\n================= 🔵 SVM =================")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

svm_model = LinearSVC()
svm_model.fit(X_train_vec, y_train)
svm_pred = svm_model.predict(X_test_vec)

print("📊 Rapport SVM :")
print(classification_report(y_test, svm_pred))

# ==============================================================================
# === MODELE 2 : LSTM ==========================================================
# ==============================================================================
print("\n================= 🔶 LSTM =================")
tokenizer = Tokenizer(num_words=5000, lower=True)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=28)
X_test_pad = pad_sequences(X_test_seq, maxlen=28)

label_to_index = {"negative": 0, "neutral": 1, "positive": 2}
index_to_label = {v: k for k, v in label_to_index.items()}
y_train_int = y_train.map(label_to_index)
y_test_int = y_test.map(label_to_index)

y_train_cat = to_categorical(y_train_int, num_classes=3)
y_test_cat = to_categorical(y_test_int, num_classes=3)

model = Sequential()
model.add(Embedding(5000, 128, input_length=28))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(196, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("🚀 Entraînement du LSTM (3 epochs)...")
model.fit(X_train_pad, y_train_cat, epochs=3, batch_size=32, verbose=2)

lstm_probs = model.predict(X_test_pad)
lstm_pred = [index_to_label[np.argmax(pred)] for pred in lstm_probs]

print("📊 Rapport LSTM :")
print(classification_report(y_test, lstm_pred))

# ==============================================================================
# === MODELE 3 : Lexicon (TextBlob) ============================================
# ==============================================================================
print("\n================= 🟢 Lexicon =================")

def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.25:
        return "positive"
    elif polarity < -0.25:
        return "negative"
    else:
        return "neutral"

lex_pred = X_test.apply(get_sentiment)

print("📊 Rapport Lexicon :")
print(classification_report(y_test, lex_pred))
