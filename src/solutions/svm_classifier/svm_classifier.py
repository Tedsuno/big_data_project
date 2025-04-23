# Generates 3 specific plots: Per-Class Metrics, Decision Scores, Text Length vs Correctness
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import joblib
from pymongo import MongoClient
from dotenv import load_dotenv
from solutions.svm_classifier.svm_plots import (
    plot_class_metrics,
    plot_decision_scores,
    plot_text_length_vs_accuracy
)

# --- Configuration ---
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://Ayoub:BigData123@bigdataproject.0fq9v2b.mongodb.net/") # Configuration to connect to the DB
DB_NAME = "sentiment_project" # The project we created inside MongoDB Atlas
COLLECTION_NAME = "tweets"
RESULTS_DIR = "results/svm" # Directory to save non-plot results
os.makedirs(RESULTS_DIR, exist_ok=True)
# ---------------------

def run_svm_classification(show_plots=True):
    """
    Loads data from MongoDB, trains SVM, evaluates, SAVES results/model,
    and optionally SHOWS 3 specific analysis plots.
    """
    start_time = time.time()

    # --- Load Data from MongoDB ---
    client = None; df = None
    try:
        print("Connecting to MongoDB Atlas..."); client = MongoClient(MONGO_URI)
        db = client[DB_NAME]; collection = db[COLLECTION_NAME] # Connection to MongoDB and Loading data
        print(f"Connected.");
        cursor = collection.find({}, {'_id': 0, 'text': 1, 'sentiment': 1}) # We delete before the others field apart from 'text' and 'sentiment'
        df = pd.DataFrame(list(cursor)); # We connect to our DB
    except Exception as e: 
        print(f"ERROR fetching data: {e}"); return
    finally:
        if client: 
           client.close(); 
           print("MongoDB connection closed.")
        if df is None or df.empty: 
           print("No data loaded. EXIT"); return

    # Basic Cleaning ( # We delete the empty entry and take only the valid sentiment )
    df.dropna(subset=['text', 'sentiment'], inplace=True) 
    valid_sentiments = ['positive', 'negative', 'neutral']
    df = df[df['sentiment'].isin(valid_sentiments)]
    df = df.reset_index(drop=True)
    print(df.shape)

    X = df['text']; y = df['sentiment']; labels = sorted(y.unique())

    # --- Split Data ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # 80% train - 20% test -- same proportion of classes
    print(f"Train={len(X_train)}, Test={len(X_test)}")

    # --- TF-IDF Vectorization --- (Vectorizing text using TF-IDF (Unigrams and Bigrams), it's essential for using SVM)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train); X_test_vec = vectorizer.transform(X_test)

    # --- Train SVM Model --- (We train a linear SVM on the vectors TF-IDF)
    model = LinearSVC(dual="auto", random_state=42, C=0.5)
    model.fit(X_train_vec, y_train)

    # --- Predictions ---
    y_pred = model.predict(X_test_vec)

    # --- Get Decision Function Scores (Needed for Plot 2) ---
    try:
        y_scores = model.decision_function(X_test_vec) # Collect the marginal scores of SVM for each classes
        if len(labels) > 2 and y_scores.ndim == 1: # Handle binary case if it occurs
             print("!!! WARNING !!!")
        elif y_scores.shape[1] != len(labels):
             print(f"!!! WARNING !!!({y_scores.shape[1]}) !!! WARNING !!! ({len(labels)})")
             y_scores = None # Cannot plot scores reliably
        else:
            print("SVM decisions score :")
    except Exception as e:
        print(f"Could not get decision scores: {e}")
        y_scores = None


    # --- Evaluation ---
    print("\n--- Classification Report ---")
    report_str = classification_report(y_test, y_pred, target_names=labels, zero_division=0)
    print(report_str)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # --- SAVING RESULTS ---
    df_preds_to_save = pd.DataFrame({'text': X_test, 'true_sentiment': y_test, 'predicted_sentiment': y_pred})
    pred_path = os.path.join(RESULTS_DIR, 'svm_predictions.csv'); df_preds_to_save.to_csv(pred_path, index=False)
    vec_path = os.path.join(RESULTS_DIR, 'tfidf_vectorizer.joblib'); joblib.dump(vectorizer, vec_path)
    model_path = os.path.join(RESULTS_DIR, 'svm_model.joblib'); joblib.dump(model, model_path)
    # ----------------------

    end_time = time.time()
    print(f"\nSVM execution time: {end_time - start_time:.2f} seconds.")

    if show_plots:
        print("\nGenerating and showing 3 analysis plots...")
        sns.set_style('whitegrid')

        print("Plot 1 : Per-Class Metrics")
        plot_class_metrics(y_test, y_pred, labels)

        print("Plot 2 : Decision Scores Distribution")
        plot_decision_scores(y_scores, y_test, labels)

        print("Plot 3 : Text Length vs Classification Accuracy")
        plot_text_length_vs_accuracy(X_test, y_test, y_pred)


if __name__ == "__main__":
     # Run the function when the script is executed directly
     run_svm_classification(show_plots=True) # Ensure plots are shown
