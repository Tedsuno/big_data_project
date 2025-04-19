# train_lstm.py
# (Imports inchangés...)
from data_preprocessing import preprocess_data # Assurez-vous que l'import est correct
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense # Ajustez les imports si nécessaire
from tensorflow.keras.preprocessing.sequence import pad_sequences
# (Autres imports...)
import numpy as np # Assurez-vous que numpy est importé


def main():
    embed_dim = 128
    lstm_out = 196
    batch_size = 32
    max_features = 5000 # Assurez-vous que c'est la même valeur que vocab_size dans preprocess_data

    print("🔁 Loading and preprocessing data FROM MONGODB...")
    # Modifier cet appel : plus de chemin CSV
    X_train, X_test, Y_train, Y_test, tokenizer = preprocess_data(max_len=28, vocab_size=max_features)

    # --- Gérer le cas où preprocess_data retourne None (erreur) ---
    if X_train is None:
         print("❌ Failed to load or preprocess data. Exiting.")
         return
    # ------------------------------------------------------------

    print("✅ Building model...")
    model = Sequential()
    model.add(Embedding(max_features, embed_dim, input_length=X_train.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    # !!! ATTENTION à la couche Dense finale et à la loss function !!!
    # Si Y_train est one-hot ([n, 2]), la dernière couche doit avoir 2 neurones et activation 'softmax'
    # Et la loss doit être 'categorical_crossentropy'
    if Y_train.shape[1] == 2:
        print("Model adapted for 2 classes (softmax output)")
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Si Y_train est binaire ([n, 1] avec 0 ou 1), la dernière couche doit avoir 1 neurone et activation 'sigmoid'
    # Et la loss doit être 'binary_crossentropy'
    elif Y_train.shape[1] == 1:
         print("Model adapted for 1 class (sigmoid output)")
         model.add(Dense(1, activation='sigmoid'))
         model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
         print(f"❌ Unexpected shape for Y_train: {Y_train.shape}. Cannot compile model.")
         return

    print(model.summary())

    print("🚀 Training...")
    # Assurez-vous que Y_train a la bonne forme pour la loss function choisie
    model.fit(X_train, Y_train, epochs=10, batch_size=batch_size, verbose=2, validation_split=0.1)

    print("📊 Evaluating...")
    # Assurez-vous que Y_test a la bonne forme
    score, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)
    print("🧾 Score: %.2f" % score)
    print("✅ Accuracy: %.2f" % acc)

    # Split manual validation
    # (Le reste du code d'évaluation semble ok, mais dépend de la forme de Y_validate)
    # ... (votre code d'évaluation existant) ...
    # Assurez-vous que np.argmax est utilisé correctement si Y_validate est [n, 2]
    # Si Y_validate est [n, 1], la comparaison sera différente.

    validation_size = 1500
    # Vérifier si assez de données dans X_test
    if len(X_test) <= validation_size:
        print(f"⚠️ Not enough samples in X_test ({len(X_test)}) for validation_size={validation_size}. Skipping detailed validation.")
    else:
        X_validate = X_test[-validation_size:]
        Y_validate = Y_test[-validation_size:]
        X_test = X_test[:-validation_size]
        Y_test = Y_test[:-validation_size] # Y_test est réduit ici

        # Analyse par classe - Adaptée pour sortie softmax/catégorique (2 classes)
        pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
        if Y_validate.shape[1] == 2: # Assumer que l'index 1 est positif, 0 est négatif
             print("🔍 Detailed validation (assuming softmax output)...")
             for i in range(len(X_validate)):
                 result = model.predict(X_validate[i].reshape(1, X_validate.shape[1]), batch_size=1, verbose=0)[0]
                 predicted_label_index = np.argmax(result)
                 true_label_index = np.argmax(Y_validate[i])

                 if predicted_label_index == true_label_index:
                     if true_label_index == 0: # Negative
                         neg_correct += 1
                     else: # Positive
                         pos_correct += 1

                 if true_label_index == 0: # Negative
                     neg_cnt += 1
                 else: # Positive
                     pos_cnt += 1

             if pos_cnt > 0: print("🎯 Positive Accuracy:", round(pos_correct / pos_cnt * 100, 2), "%")
             else: print("🎯 No positive samples in validation set.")
             if neg_cnt > 0: print("🎯 Negative Accuracy:", round(neg_correct / neg_cnt * 100, 2), "%")
             else: print("🎯 No negative samples in validation set.")
        else:
             print("⚠️ Detailed validation requires adaptation for non-categorical output.")


    # Test rapide - Adapté pour sortie softmax
    print("\n📍 Test sur un tweet :")
    twt = ['Meetings: Because none of us is as dumb as all of us.']
    twt_seq = tokenizer.texts_to_sequences(twt)
    twt_pad = pad_sequences(twt_seq, maxlen=28) # Utiliser la même max_len
    prediction = model.predict(twt_pad, batch_size=1, verbose=0)[0]

    if Y_train.shape[1] == 2: # Softmax
        label_index = np.argmax(prediction)
        # Suppose que l'index 0 correspond à 'negative' et l'index 1 à 'positive'
        # Ceci dépend de l'ordre des colonnes dans pd.get_dummies(df['sentiment'])
        # Vérifiez cet ordre si besoin (ex: print(pd.get_dummies(df['sentiment']).columns))
        label = "negative" if label_index == 0 else "positive" # Ajustez si nécessaire
        confidence = prediction[label_index] * 100
        print(f"Tweet: {twt[0]}")
        print(f"Predicted sentiment: {label} ({confidence:.2f}%)")
    else: # Sigmoid
        # Supposons que la sortie > 0.5 est positive
        confidence = prediction[0] * 100
        label = "positive" if prediction[0] > 0.5 else "negative"
        print(f"Tweet: {twt[0]}")
        print(f"Predicted sentiment: {label} ({confidence:.2f}%)")


if __name__ == "__main__":
    main()