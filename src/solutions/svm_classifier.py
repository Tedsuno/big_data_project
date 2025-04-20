# Modified src/solutions/svm_classifier.py
# Generates 3 specific plots: Per-Class Metrics, Decision Scores, Text Length vs Correctness

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix # Added confusion_matrix back just in case
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import joblib
from pymongo import MongoClient
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://Ayoub:BigData123@bigdataproject.0fq9v2b.mongodb.net/")
DB_NAME = "sentiment_project"
COLLECTION_NAME = "tweets"
RESULTS_DIR = "results/svm" # Directory to save non-plot results
os.makedirs(RESULTS_DIR, exist_ok=True)
# ---------------------

def run_svm_classification(show_plots=True):
    """
    Loads data from MongoDB, trains SVM, evaluates, SAVES results/model,
    and optionally SHOWS 3 specific analysis plots.
    """
    print("--- Running SVM Classification (MongoDB -> Save Results -> Show 3 Plots) ---")
    start_time = time.time()

    # --- Load Data from MongoDB ---
    client = None; df = None
    try:
        print("🌍 Connecting to MongoDB Atlas..."); client = MongoClient(MONGO_URI)
        db = client[DB_NAME]; collection = db[COLLECTION_NAME]
        print(f"✅ Connected."); print("⬇️ Fetching data (text, sentiment)...")
        cursor = collection.find({}, {'_id': 0, 'text': 1, 'sentiment': 1})
        df = pd.DataFrame(list(cursor)); print(f"📊 Fetched {len(df)} documents.")
    except Exception as e: print(f"❌ ERROR fetching data: {e}"); return
    finally:
        if client: client.close(); print("🔒 MongoDB connection closed.")
    if df is None or df.empty: print("❌ No data loaded. Exiting."); return

    # Basic Cleaning
    df.dropna(subset=['text', 'sentiment'], inplace=True)
    valid_sentiments = ['positive', 'negative', 'neutral']
    df = df[df['sentiment'].isin(valid_sentiments)]; df = df.reset_index(drop=True)
    print(f"Data shape after cleaning: {df.shape}")

    X = df['text']; y = df['sentiment']; labels = sorted(y.unique())

    # --- Split Data ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Data split: Train={len(X_train)}, Test={len(X_test)}")

    # --- TF-IDF Vectorization ---
    print("Vectorizing text using TF-IDF (Unigrams & Bigrams)...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train); X_test_vec = vectorizer.transform(X_test)
    print("Vectorization complete.")

    # --- Train SVM Model ---
    print("Training LinearSVC model...")
    # Assuming OvR multi-class strategy (default for LinearSVC)
    model = LinearSVC(dual="auto", random_state=42, C=0.5)
    model.fit(X_train_vec, y_train)
    print("Training complete.")

    # --- Predictions ---
    print("Making predictions...")
    y_pred = model.predict(X_test_vec)

    # --- Get Decision Function Scores (Needed for Plot 2) ---
    try:
        y_scores = model.decision_function(X_test_vec)
        if len(labels) > 2 and y_scores.ndim == 1: # Handle binary case if it occurs
             print("Warning: decision_function returned 1D array for multi-class, might need score adjustment.")
             # Create dummy scores for plotting if needed, or adjust logic
        elif y_scores.shape[1] != len(labels):
             print(f"Warning: Mismatch between score columns ({y_scores.shape[1]}) and labels ({len(labels)})")
             y_scores = None # Cannot plot scores reliably
        else:
            print("Generated SVM decision scores.")
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
    print("\n💾 Saving results and model components...")
    df_preds_to_save = pd.DataFrame({'text': X_test, 'true_sentiment': y_test, 'predicted_sentiment': y_pred})
    pred_path = os.path.join(RESULTS_DIR, 'svm_predictions.csv'); df_preds_to_save.to_csv(pred_path, index=False)
    print(f"   -> Predictions saved to: {pred_path}")
    vec_path = os.path.join(RESULTS_DIR, 'tfidf_vectorizer.joblib'); joblib.dump(vectorizer, vec_path)
    print(f"   -> Vectorizer saved to: {vec_path}")
    model_path = os.path.join(RESULTS_DIR, 'svm_model.joblib'); joblib.dump(model, model_path)
    print(f"   -> SVM model saved to: {model_path}")
    # ----------------------

    end_time = time.time()
    print(f"\n--- SVM execution time: {end_time - start_time:.2f} seconds ---")

    # --- Plotting Section (3 Specific Plots, Show Directly) ---
    if show_plots:
        print("\n📊 Generating and showing 3 analysis plots...")
        sns.set_style('whitegrid')

        # === Plot 1: Precision, Recall, F1-Score per Class ===
        print("   -> Plot 1: Generating Per-Class Metrics (P, R, F1)...")
        try:
            report_dict = classification_report(y_test, y_pred, target_names=labels, output_dict=True, zero_division=0)
            metrics_to_plot = ['precision', 'recall', 'f1-score']
            class_metrics = {}
            for label in labels:
                 # Ensure label exists in report (might not if a class has 0 support in test)
                 if label in report_dict:
                      class_metrics[label] = {metric: report_dict[label][metric] for metric in metrics_to_plot}

            if class_metrics: # Check if we have metrics to plot
                df_metrics = pd.DataFrame(class_metrics).T.reset_index().rename(columns={'index': 'Sentiment'})
                df_melted_metrics = df_metrics.melt(id_vars='Sentiment', var_name='Metric', value_name='Score')

                plt.figure(figsize=(10, 6))
                ax = sns.barplot(data=df_melted_metrics, x='Sentiment', y='Score', hue='Metric', palette='cubehelix')
                plt.title('SVM: Precision, Recall, F1-Score per Class')
                plt.xlabel('Sentiment Class'); plt.ylabel('Score'); plt.ylim(0, 1.05)
                plt.legend(title='Metric', bbox_to_anchor=(1.02, 1), loc='upper left')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                # Add score labels
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.2f', label_type='edge', padding=2, fontsize=9)
                plt.tight_layout(); plt.show()
            else:
                 print("      Skipping plot: No class metrics found in report.")
        except Exception as e:
            print(f"      Error generating per-class metrics plot: {e}")


        # === Plot 2: Distribution of SVM Decision Scores per True Class ===
        print("\n   -> Plot 2: Generating Decision Score Distributions...")
        if y_scores is not None and y_scores.shape[1] == len(labels):
            try:
                df_scores = pd.DataFrame(y_scores, columns=[f"score_{l}" for l in labels])
                df_scores['true_sentiment'] = y_test.values # Add true labels

                plt.figure(figsize=(12, 7))
                for i, label in enumerate(labels):
                    # Plot the distribution of scores assigned *to this class's dimension*...
                    # ...for samples truly belonging to this class vs samples belonging to other classes
                    sns.kdeplot(data=df_scores, x=f"score_{label}", hue='true_sentiment', fill=True, common_norm=False, alpha=0.5)
                    plt.title(f"Distribution of SVM Decision Scores for Class '{label}' Dimension")
                    plt.xlabel(f"Decision Score (Dimension: {label})")
                    plt.ylabel("Density")
                    plt.grid(True, linestyle='--', alpha=0.6)
                    plt.show() # Show one plot per class dimension

            except Exception as e:
                print(f"      Error generating decision score plot: {e}")
        else:
            print("      Skipping decision score plot: Scores not available or shape mismatch.")


        # === Plot 3: Text Length Distribution for Correct vs. Incorrect Predictions ===
        print("\n   -> Plot 3: Generating Text Length Distribution (Correct vs. Incorrect)...")
        try:
            # Ensure X_test is available (it should be from the split above)
            if 'X_test' in locals():
                text_lengths = X_test.astype(str).apply(len)
                is_correct = (y_test.values == y_pred) # Ensure comparison is valid

                df_plot_len = pd.DataFrame({
                    'text_length': text_lengths,
                    'correctly_classified': is_correct
                })

                plt.figure(figsize=(12, 6))
                sns.histplot(data=df_plot_len, x='text_length', hue='correctly_classified', kde=True, palette={True: 'forestgreen', False: 'firebrick'}) # Changed palette
                plt.title('Distribution of Text Length for Correct vs. Incorrect SVM Predictions')
                plt.xlabel('Text Length (Characters)'); plt.ylabel('Frequency')
                plt.legend(title='Correctly Classified', labels=['Correct', 'Incorrect']) # Match hue order/palette
                plt.tight_layout(); plt.show()
            else:
                print("      Skipping text length plot: X_test not found.")
        except Exception as e:
            print(f"      Error generating text length plot: {e}")


    print("\n--- SVM Script Finished ---")


if __name__ == "__main__":
     # Run the function when the script is executed directly
     run_svm_classification(show_plots=True) # Ensure plots are shown