import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np

def plot_training_history(history):
    """Affiche les courbes de loss et d'accuracy sur les epochs."""
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train acc')
    plt.plot(history.history['val_accuracy'], label='Val acc')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Val loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_wordcloud_from_texts(texts, title):
    """Génère un nuage de mots à partir d'une liste de textes."""
    text = " ".join(texts)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    plt.show()

def analyze_misclassified(model, X_data, Y_data, texts, tokenizer):
    """Affiche les nuages de mots des bien/mal classés."""
    predictions = model.predict(X_data, verbose=0)
    y_true = np.argmax(Y_data, axis=1)
    y_pred = np.argmax(predictions, axis=1)

    misclassified_texts = [texts[i] for i in range(len(y_true)) if y_true[i] != y_pred[i]]
    correct_texts = [texts[i] for i in range(len(y_true)) if y_true[i] == y_pred[i]]

    print(f"Total misclassified: {len(misclassified_texts)}")
    print(f"Total correctly classified: {len(correct_texts)}")

    plot_wordcloud_from_texts(misclassified_texts, "Figure 9. Frequent words in tweets misclassified by the LSTM model")
    plot_wordcloud_from_texts(correct_texts, "Frequent words in correctly classified tweets")
