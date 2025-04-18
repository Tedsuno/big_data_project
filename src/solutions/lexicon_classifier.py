import pandas as pd
from textblob import TextBlob
import re


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
    df = pd.read_csv(csv_file)

    # Check if necessary columns exist in the dataframe
    if not {'textID', 'text', 'selected_text', 'sentiment'}.issubset(df.columns):
        raise ValueError("CSV file must contain 'textID', 'text', 'selected_text', and 'sentiment' columns")

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


# Main entry point
if __name__ == "__main__":
    file_path = "../../data/tweet.csv"  # Change this to the actual CSV file path
    result_df = analyze_tweets(file_path)
    print(result_df.head())  # Print the first few rows of the result