import re

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s,]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()
