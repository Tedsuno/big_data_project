from textblob import TextBlob
import pandas as pd

def get_sentiment(text, threshold=0.25):
    if pd.isna(text):
        return "neutral"
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > threshold:
        return "positive"
    elif polarity < -threshold:
        return "negative"
    return "neutral"
