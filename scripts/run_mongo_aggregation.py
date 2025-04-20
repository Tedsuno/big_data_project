# scripts/run_mongo_aggregation.py
import os
import pprint # For nice printing of results
from pymongo import MongoClient
from dotenv import load_dotenv # To load credentials safely (optional but recommended)

# --- Configuration ---
# Load environment variables from .env file if it exists (recommended for credentials)
load_dotenv()

# Get MongoDB URI from environment variable or use the hardcoded one (less secure)
# Create a file named '.env' in your project root with: MONGO_URI="mongodb+srv://..."
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://Ayoub:BigData123@bigdataproject.0fq9v2b.mongodb.net/")
DB_NAME = "sentiment_project"
COLLECTION_NAME = "tweets"
# ---------------------


def analyze_sentiment_stats():
    """
    Connects to MongoDB and runs an aggregation pipeline to calculate
    count and average text length per sentiment.
    """
    client = None
    try:
        print("🌍 Connecting to MongoDB Atlas...")
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        print(f"✅ Connected to DB: '{DB_NAME}', Collection: '{COLLECTION_NAME}'")

        print("\n⚙️ Running Aggregation Pipeline...")

        # --- Aggregation Pipeline Definition ---
        pipeline = [
            {
                # Stage 1: Filter out documents with missing 'text' or 'sentiment' (optional, good practice)
                '$match': {
                    'text': {'$exists': True, '$ne': None, '$ne': ''},
                    'sentiment': {'$exists': True, '$in': ['positive', 'negative', 'neutral']}
                }
            },
            {
                # Stage 2: Add a field for the length of the 'text' string
                '$addFields': {
                    'text_length': {'$strLenCP': '$text'} # Calculates string length in bytes (UTF-8 chars)
                }
            },
            {
                # Stage 3: Group by sentiment and calculate count + average length
                '$group': {
                    '_id': '$sentiment',  # Group by the sentiment field
                    'count': {'$sum': 1}, # Count documents in each group
                    'average_text_length': {'$avg': '$text_length'} # Calculate average of the new field
                }
            },
            {
                # Stage 4: Project to rename fields for nicer output (optional)
                '$project': {
                    '_id': 0, # Exclude the default _id field
                    'sentiment': '$_id', # Rename _id to sentiment
                    'tweet_count': '$count',
                    'avg_text_length': {'$round': ['$average_text_length', 2]} # Round avg length
                }
            },
            {
                # Stage 5: Sort results by sentiment name (optional)
                 '$sort': {'sentiment': 1}
            }
        ]
        # -----------------------------------------

        # Execute the aggregation pipeline
        results = list(collection.aggregate(pipeline))

        print("\n--- Aggregation Results ---")
        if results:
            pprint.pprint(results)
        else:
            print("No results found. Check if the collection has data matching the criteria.")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
    finally:
        if client:
            client.close()
            print("\n🔒 MongoDB connection closed.")

# Main execution block
if __name__ == "__main__":
    analyze_sentiment_stats()