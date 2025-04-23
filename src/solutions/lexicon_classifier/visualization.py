import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sentiment_analyzer import get_sentiment


def visualize_sentiment_analysis(df):
    """
    Generate visualizations to evaluate sentiment analysis performance.
    Assumes df has columns: 'sentiment', 'predicted_sentiment', 'cleaned_text'
    """
    sns.set(style="whitegrid", palette="muted", font_scale=1.1)

    # --- 1. Confusion Matrix ---
    plt.figure(figsize=(6, 5))
    labels = ['positive', 'neutral', 'negative']
    cm = confusion_matrix(df['sentiment'], df['predicted_sentiment'], labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Sentiment')
    plt.ylabel('Actual Sentiment')
    plt.tight_layout()
    plt.show()

    # --- 2. Sentiment Distribution ---
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    sns.countplot(data=df, x='sentiment', order=labels)
    plt.title('Actual Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    sns.countplot(data=df, x='predicted_sentiment', order=labels)
    plt.title('Predicted Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show()

    # --- 3. Accuracy vs Threshold ---
    thresholds = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    accuracy_scores = []

    for t in thresholds:
        predicted = df['cleaned_text'].apply(lambda x: get_sentiment(x, threshold=t))
        accuracy = (predicted == df['sentiment']).mean()
        accuracy_scores.append(accuracy)

    # Plot threshold comparison
    plt.figure(figsize=(10, 6))
    bars = sns.barplot(x=thresholds, y=accuracy_scores, palette='Greens_d')

    # Highlight best threshold
    best_idx = max(range(len(accuracy_scores)), key=lambda i: accuracy_scores[i])
    bars.patches[best_idx].set_color('crimson')
    plt.text(best_idx, accuracy_scores[best_idx] + 0.01,
             f"{accuracy_scores[best_idx]:.2%}", ha='center',
             va='bottom', fontsize=10, fontweight='bold', color='crimson')

    plt.title("Accuracy vs Sentiment Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.grid(axis='y')
    plt.show()
