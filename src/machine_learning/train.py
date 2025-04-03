from data_preprocessing import preprocess_data
from model_lstm import create_lstm_model
from sklearn.metrics import classification_report
import numpy as np

def main():
    print("🧼 Preprocessing data...")
    X_train, X_test, y_train, y_test, tokenizer = preprocess_data("../../data/tweet.csv")

    vocab_size = len(tokenizer.word_index) + 1
    input_length = X_train.shape[1]

    print("🧠 Creating model...")
    model = create_lstm_model(vocab_size, input_length)

    print("🏋️ Training model...")
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

    print("📊 Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_binary = np.round(y_pred)

    print("\n🧾 Classification report:")
    print(classification_report(y_test, y_pred_binary, target_names=["non-haine", "haine"]))

    accuracy = np.mean(y_pred_binary.flatten() == y_test)
    print(f"\n✅ Prediction accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
