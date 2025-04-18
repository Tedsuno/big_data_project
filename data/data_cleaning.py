import pandas as pd
import re

# Your preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s,]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text

# Load the CSV
df = pd.read_csv("/Users/jonasebert/PycharmProjects/big_data_project/data/tweet.csv")

# Apply preprocessing to the 'text' column (and optionally 'selected_text')
df['cleaned_text'] = df['text'].apply(preprocess_text)
df['cleaned_selected_text'] = df['selected_text'].apply(preprocess_text)

# Save to a new CSV
df.to_csv("cleaned_data.csv", index=False)

print("Cleaned data saved to cleaned_data.csv ✅")