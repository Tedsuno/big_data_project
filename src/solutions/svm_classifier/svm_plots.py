import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

def plot_class_metrics(y_test, y_pred, labels):
    report_dict = classification_report(y_test, y_pred, target_names=labels, output_dict=True, zero_division=0)
    metrics_to_plot = ['precision', 'recall', 'f1-score']
    class_metrics = {
        label: {metric: report_dict[label][metric] for metric in metrics_to_plot}
        for label in labels if label in report_dict
    }


    if class_metrics:
        df = pd.DataFrame(class_metrics).T.reset_index().rename(columns={'index': 'Sentiment'})
        df_melt = df.melt(id_vars='Sentiment', var_name='Metric', value_name='Score')

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=df_melt, x='Sentiment', y='Score', hue='Metric', palette='cubehelix')
        plt.title('SVM: Precision, Recall, F1-Score per Class')
        plt.xlabel('Sentiment Class'); plt.ylabel('Score'); plt.ylim(0, 1.05)
        ax.legend(title='Metric', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', label_type='edge', padding=2, fontsize=9)
        plt.tight_layout(); plt.show()
    else:
        print(" !!! ERROR1 !!!")

def plot_decision_scores(y_scores, y_test, labels):
    if y_scores is None or y_scores.shape[1] != len(labels):
        print(" !!! ERROR2 !!!")
        return

    df_scores = pd.DataFrame(y_scores, columns=[f"score_{l}" for l in labels])
    df_scores['true_sentiment'] = y_test.values

    for label in labels:
        plt.figure(figsize=(12, 7))
        sns.kdeplot(data=df_scores, x=f"score_{label}", hue='true_sentiment', fill=True, common_norm=False, alpha=0.5)
        plt.title(f"Distribution of SVM Decision Scores for Class '{label}' Dimension")
        plt.xlabel(f"Decision Score (Dimension: {label})")
        plt.ylabel("Density")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout(); plt.show()

def plot_text_length_vs_accuracy(X_test, y_test, y_pred):
    text_lengths = X_test.astype(str).apply(len)
    is_correct = (y_test.values == y_pred)

    df_plot = pd.DataFrame({
        'text_length': text_lengths,
        'correctly_classified': is_correct
    })

    plt.figure(figsize=(12, 6))
    sns.histplot(data=df_plot, x='text_length', hue='correctly_classified', kde=True,
                 palette={True: 'forestgreen', False: 'firebrick'}, alpha=0.6)
    plt.title('Distribution of Text Length for Correct vs. Incorrect SVM Predictions')
    plt.xlabel('Text Length (Characters)')
    plt.ylabel('Frequency')
    plt.legend(title='Correctly Classified', labels=['Correct', 'Incorrect'])
    plt.tight_layout(); plt.show()
