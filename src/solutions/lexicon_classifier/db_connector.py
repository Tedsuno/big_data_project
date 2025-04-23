import pandas as pd
from pymongo import MongoClient
from config import MONGO_URI, DB_NAME, COLLECTION_NAME

def fetch_tweets():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    cursor = collection.find({}, {'_id': 0, 'text': 1, 'sentiment': 1, 'selected_text': 1, 'textID': 1})
    return pd.DataFrame(list(cursor))
