# train_lstm.py
# (Imports)
from data_preprocessing import preprocess_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from model_lstm import create_lstm_model
from plot import plot_training_history, analyze_misclassified

def print_results(validation_size, X_train, X_test, Y_train, Y_test, tokenizer,model) :
    if len(X_test) <= validation_size:
        print(f" Not enough samples")
    else:
        X_validate, Y_validate = X_test[-validation_size:], Y_test[-validation_size:]
        X_test, Y_test = X_test[:-validation_size], Y_test[:-validation_size]

        if Y_validate.shape[1] == 2:  # Sortie softmax
            print("Detailed validation (softmax output)...")
            
            
            y_true = np.argmax(Y_validate, axis=1)
            y_pred = np.argmax(model.predict(X_validate, verbose=0), axis=1)

            for label, name in zip([1, 0], ["Positive", "Negative"]):
                mask = y_true == label
                if mask.any():
                    acc = np.mean(y_pred[mask] == label)
                    print(f"{name} Accuracy: {acc * 100:.2f}%")
                else:
                    print(f"No {name.lower()} samples in validation set.")
        else: # Cas sigmoid
            print("Detailed validation requires adaptation for non-categorical output.")

        # Prédiction sur un tweet exemple
        print("\n Test sur un tweet :")
        twt = ['Meetings: Because none of us is as dumb as all of us.']
        twt_pad = pad_sequences(tokenizer.texts_to_sequences(twt), maxlen=28)
        prediction = model.predict(twt_pad, verbose=0)[0]

        if Y_train.shape[1] == 2:
            label_index = np.argmax(prediction)
            label = "positive" if label_index == 1 else "negative"
            confidence = prediction[label_index] * 100
        else:
            confidence = prediction[0] * 100
            label = "positive" if prediction[0] > 0.5 else "negative"

    print(f"Tweet: {twt[0]}")
    print(f"Predicted sentiment: {label} ({confidence:.2f}%)")

# (Main)
def main():
    batch_size = 32
    max_features = 5000
    max_len = 28

    print("Chargement et prétraitement UNIQUE des données et création du modèle LSTM...")

    X_train, X_test, Y_train, Y_test, texts_train, texts_test, tokenizer = preprocess_data(
        max_len=max_len,
        vocab_size=max_features,
        test_size=0.33,
        random_state=42
    )
    if X_train is None:
         print("Le prétraitement des données a échoué. Arrêt.")
         return

    # Création du modèle LSTM
    model = create_lstm_model(max_features, X_train.shape[1], Y_train, X_train)
    #Entrainement du modele
    print("Entrainement")
    history = model.fit(X_train, Y_train, epochs=10, batch_size=batch_size, verbose=2, validation_split=0.1)
    #Evaluation de l'accuracy
    print("Evaluation")
    score, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)
    print(f"Score: {score:.2f}")
    print(f"Accuracy: {acc:.2f}")
    #Affichage des résulats numériques
    validation_size = 1500
    print_results(validation_size, X_train, X_test, Y_train, Y_test, tokenizer, model)

    # Affichage des graphiques
    plot_training_history(history) 
    analyze_misclassified(model, X_test, Y_test, texts_test, tokenizer) 


if __name__ == "__main__":
    main()
