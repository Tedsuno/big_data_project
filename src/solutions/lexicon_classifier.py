import pandas as pd
#import certifi
from textblob import TextBlob
import re
from pymongo import MongoClient
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

MONGO_URI = "mongodb+srv://Ayoub:BigData123@bigdataproject.0fq9v2b.mongodb.net/"
DB_NAME = "sentiment_project"
COLLECTION_NAME = "tweets"


# Function to preprocess the text (clean the tweet)
def preprocess_text(text):
    # Ensure the input is a valid string (if it's not, return an empty string or placeholder)
    if not isinstance(text, str):
        return ""  # Return an empty string if text is not a valid string

    # Remove URLs (e.g., http:// or www.)
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove mentions (@username) and hashtags (#hashtag)
    text = re.sub(r'@\w+|#\w+', '', text)

    # Remove special characters and punctuation (optional step for cleaning)
    text = re.sub(r'[^\w\s,]', '', text)

    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Convert to lowercase
    text = text.lower()

    return text


# Function to classify sentiment as positive, neutral, or negative based on polarity
def get_sentiment(text):
    if pd.isna(text):  # Handle missing or NaN values
        return "neutral"  # Default to neutral if text is missing
    analysis = TextBlob(str(text))  # Ensure text is converted to string
    polarity = analysis.sentiment.polarity

    # Classify based on polarity score
    if polarity > 0.25:
        return "positive"
    elif polarity < -0.25:
        return "negative"
    else:
        return "neutral"


# Function to analyze tweets from CSV file and calculate accuracy
def analyze_tweets(csv_file):

    print("🌍 Connecting to MongoDB Atlas...")
    # Se connecter à la base de données
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    print(f"✅ Connected to DB: '{DB_NAME}', Collection: '{COLLECTION_NAME}'")

    print("⬇️ Fetching data from MongoDB...")
    cursor = collection.find({}, {'_id': 0, 'text': 1, 'sentiment': 1, 'selected_text' : 1, 'textID' : 1 })

    # Convertir les données récupérées en DataFrame pandas
    df = pd.DataFrame(list(cursor))

    # Preprocess the tweets (cleaning the text)
    df['cleaned_text'] = df['selected_text'].apply(preprocess_text)

    # Perform sentiment analysis on the cleaned 'selected_text' column
    df['predicted_sentiment'] = df['cleaned_text'].apply(get_sentiment)

    # Calculate accuracy by comparing predicted sentiment with actual sentiment
    correct_predictions = (df['predicted_sentiment'] == df['sentiment']).sum()
    total_predictions = len(df)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Sentiment Analysis Accuracy: {accuracy:.2%}")

    # Return dataframe with relevant columns
    return df[['textID', 'text', 'selected_text', 'sentiment', 'predicted_sentiment']]

def visualize_sentiment_analysis(df):
    """
    Generate visualizations to evaluate sentiment analysis performance.
    Assumes df has columns: 'sentiment', 'predicted_sentiment', 'cleaned_text'
    """
    # Set style
    sns.set(style="whitegrid", palette="muted", font_scale=1.1)

    # 1. Confusion Matrix
    plt.figure(figsize=(6, 5))
    labels = ['positive', 'neutral', 'negative']
    cm = confusion_matrix(df['sentiment'], df['predicted_sentiment'], labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Sentiment')
    plt.ylabel('Actual Sentiment')
    plt.tight_layout()
    plt.show()

    # 2. Sentiment Distribution (Bar Plot)
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


# Main entry point
if __name__ == "__main__":
    file_path = "../../data/tweet.csv"  # Change this to the actual CSV file path
    result_df = analyze_tweets(file_path)
    print(result_df.head())  # Print the first few rows of the result
    visualize_sentiment_analysis(result_df)