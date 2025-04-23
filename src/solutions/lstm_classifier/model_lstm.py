from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from data_preprocessing import preprocess_data

def create_lstm_model(vocab_size, input_length,Y_train,X_train):
    embed_dim = 128
    lstm_out = 196
    max_features = 5000
    print("✅ Building model...")
    model = Sequential()
    model.add(Embedding(max_features, embed_dim, input_length=X_train.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    if Y_train.shape[1] == 2:
        print("Model adapted for 2 classes (softmax output)")
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif Y_train.shape[1] == 1:
         print("Model adapted for 1 class (sigmoid output)")
         model.add(Dense(1, activation='sigmoid'))
         model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
         print(f"❌ Unexpected shape for Y_train: {Y_train.shape}. Cannot compile model.")
         return

    print(model.summary())
    
    
    return model
