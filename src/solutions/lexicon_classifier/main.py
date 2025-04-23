from db_connector import fetch_tweets
from preprocessing import preprocess_text
from sentiment_analyzer import get_sentiment
from visualization import visualize_sentiment_analysis

if __name__ == "__main__":
    df = fetch_tweets()
    df['cleaned_text'] = df['selected_text'].apply(preprocess_text)
    df['predicted_sentiment'] = df['cleaned_text'].apply(get_sentiment)

    accuracy = (df['predicted_sentiment'] == df['sentiment']).mean()
    print(f"🎯 Accuracy: {accuracy:.2%}")

    visualize_sentiment_analysis(df)
