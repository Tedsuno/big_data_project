from data_preprocessing import preprocess_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense
from sklearn.metrics import classification_report
import numpy as np

def main():
    embed_dim = 128
    lstm_out = 196
    batch_size = 32
    max_features = 5000

    print("🔁 Loading and preprocessing data...")
    X_train, X_test, Y_train, Y_test, tokenizer = preprocess_data("../../data/tweet.csv", max_len=28, vocab_size=max_features)

    print("✅ Building model...")
    model = Sequential()
    model.add(Embedding(max_features, embed_dim, input_length=X_train.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='softmax'))  # 2 classes
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())

    print("🚀 Training...")
    model.fit(X_train, Y_train, epochs=10, batch_size=batch_size, verbose=2, validation_split=0.1)

    print("📊 Evaluating...")
    score, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)
    print("🧾 Score: %.2f" % score)
    print("✅ Accuracy: %.2f" % acc)

    # Split manual validation
    validation_size = 1500
    X_validate = X_test[-validation_size:]
    Y_validate = Y_test[-validation_size:]
    X_test = X_test[:-validation_size]
    Y_test = Y_test[:-validation_size]

    # Analyse par classe
    pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
    for i in range(len(X_validate)):
        result = model.predict(X_validate[i].reshape(1, X_validate.shape[1]), batch_size=1, verbose=0)[0]
        if np.argmax(result) == np.argmax(Y_validate[i]):
            if np.argmax(Y_validate[i]) == 0:
                neg_correct += 1
            else:
                pos_correct += 1
        if np.argmax(Y_validate[i]) == 0:
            neg_cnt += 1
        else:
            pos_cnt += 1

    print("🎯 Positive Accuracy:", round(pos_correct / pos_cnt * 100, 2), "%")
    print("🎯 Negative Accuracy:", round(neg_correct / neg_cnt * 100, 2), "%")

    # Test rapide
    print("\n📍 Test sur un tweet :")
    twt = ['Meetings: Because none of us is as dumb as all of us.']
    twt_seq = tokenizer.texts_to_sequences(twt)
    twt_pad = pad_sequences(twt_seq, maxlen=28)
    prediction = model.predict(twt_pad, batch_size=1, verbose=0)[0]
    label = "negative" if np.argmax(prediction) == 0 else "positive"
    print(f"Tweet: {twt[0]}")
    print(f"Predicted sentiment: {label}")

if __name__ == "__main__":
    main()
