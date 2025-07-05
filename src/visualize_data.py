import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from wordcloud import WordCloud
from collections import Counter
import re
import emoji
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
import os

# Initialize spaCy with efficient settings
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

# Text cleaning function matching traditional.py
def clean_text(text):
    """Text cleaning identical to traditional.py"""
    text = text.lower()
    text = re.sub(r'https?://\S+', ' <URL> ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'@\w+|#\w+', ' ', text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r'[^a-z0-9<>\[\]_.,!?\s]', ' ', text)
    text = " ".join(text.split()).strip()
    return text if text else "[BLANK]"

# Load data with progress indicator
def load_data(train_path, test_path):
    """Load JSONL files with progress tracking"""
    print("Loading train data...")
    train_df = pd.read_json(train_path, lines=True)
    print("Loading test data...")
    test_df = pd.read_json(test_path, lines=True)
    return train_df, test_df

# Preprocess data efficiently
def preprocess_data(df):
    """Add cleaned text and features with batch processing"""
    df = df.copy()
    print("Cleaning text...")
    df['text_clean'] = df['text'].apply(clean_text)
    
    print("Calculating basic features...")
    df['char_count'] = df['text_clean'].str.len()
    df['word_count'] = df['text_clean'].str.split().str.len()
    
    print("Processing with spaCy (this may take a while)...")
    # Batch processing for spaCy features
    docs = nlp.pipe(tqdm(df['text_clean'], total=len(df)), batch_size=128)
    sentence_counts = []
    avg_word_lengths = []
    
    for doc in docs:
        words = [token.text for token in doc if not token.is_punct and not token.is_space]
        sentence_counts.append(len(list(doc.sents)))
        avg_word_lengths.append(np.mean([len(w) for w in words]) if words else 0)
    
    df['sentence_count'] = sentence_counts
    df['avg_word_length'] = avg_word_lengths
    df['avg_sentence_length'] = df['word_count'] / df['sentence_count'].replace(0, 1)
    
    return df

# Enhanced visualization functions
def plot_class_distribution(train_df, test_df, id2label):
    """Plot class distribution with percentages"""
    fig, ax = plt.subplots(1, 2, figsize=(18, 7))
    
    # Train distribution
    train_counts = train_df['label'].value_counts().sort_index()
    sns.barplot(x=train_counts.index, y=train_counts.values, ax=ax[0], palette="viridis")
    ax[0].set_title('Train Set Class Distribution', fontsize=14)
    ax[0].set_xlabel('Class')
    ax[0].set_ylabel('Count')
    ax[0].set_xticklabels([id2label[i] for i in train_counts.index], rotation=45)
    
    # Add percentages
    total = len(train_df)
    for i, v in enumerate(train_counts):
        ax[0].text(i, v + 0.01*total, f'{v/total:.1%}', ha='center', fontsize=10)
    
    # Test distribution
    test_counts = test_df['label'].value_counts().sort_index()
    sns.barplot(x=test_counts.index, y=test_counts.values, ax=ax[1], palette="viridis")
    ax[1].set_title('Test Set Class Distribution', fontsize=14)
    ax[1].set_xlabel('Class')
    ax[1].set_ylabel('Count')
    ax[1].set_xticklabels([id2label[i] for i in test_counts.index], rotation=45)
    
    # Add percentages
    total = len(test_df)
    for i, v in enumerate(test_counts):
        ax[1].text(i, v + 0.01*total, f'{v/total:.1%}', ha='center', fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_text_length_distribution(train_df, test_df):
    """Compare text length distributions with KDE"""
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    
    # Character count distribution
    sns.histplot(train_df['char_count'], bins=50, kde=True, ax=ax[0], 
                 color='skyblue', alpha=0.7, stat='density')
    ax[0].set_title('Character Count Distribution (Train)', fontsize=14)
    ax[0].set_xlabel('Character Count')
    ax[0].set_ylabel('Density')
    ax[0].set_xlim(0, 2000)
    
    sns.histplot(test_df['char_count'], bins=50, kde=True, ax=ax[1], 
                 color='salmon', alpha=0.7, stat='density')
    ax[1].set_title('Character Count Distribution (Test)', fontsize=14)
    ax[1].set_xlabel('Character Count')
    ax[1].set_ylabel('Density')
    ax[1].set_xlim(0, 2000)
    
    plt.tight_layout()
    return fig

def plot_feature_distributions(train_df, test_df):
    """Compare feature distributions between train and test"""
    features = ['word_count', 'sentence_count', 'avg_word_length', 'avg_sentence_length']
    titles = ['Word Count', 'Sentence Count', 'Average Word Length', 'Average Sentence Length']
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        # Combine train and test for better KDE estimation
        sns.kdeplot(train_df[feature], label='Train', ax=axes[i], fill=True, 
                    color='skyblue', alpha=0.5, common_norm=False)
        sns.kdeplot(test_df[feature], label='Test', ax=axes[i], fill=True, 
                    color='salmon', alpha=0.5, common_norm=False)
        
        axes[i].set_title(f'Distribution of {titles[i]}', fontsize=14)
        axes[i].set_xlabel(titles[i])
        axes[i].set_ylabel('Density')
        axes[i].legend()
        axes[i].grid(alpha=0.3)
        
        # Set reasonable limits
        if feature == 'word_count':
            axes[i].set_xlim(0, 300)
        elif feature == 'sentence_count':
            axes[i].set_xlim(0, 30)
        elif feature == 'avg_word_length':
            axes[i].set_xlim(2, 10)
        elif feature == 'avg_sentence_length':
            axes[i].set_xlim(0, 30)
    
    plt.tight_layout()
    return fig

def plot_pos_distribution(train_df, id2label, sample_size=500):
    """Visualize POS tag distribution per class"""
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 2, figure=fig)
    
    classes = sorted(train_df['label'].unique())
    pos_tags = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'ADP', 'CONJ', 'DET', 'NUM', 'PART', 'X']
    
    for i, cls in enumerate(classes):
        ax = fig.add_subplot(gs[i])
        class_sample = train_df[train_df['label'] == cls].sample(min(sample_size, len(train_df)))
        
        pos_counts = {tag: 0 for tag in pos_tags}
        docs = nlp.pipe(class_sample['text_clean'], batch_size=64)
        
        for doc in docs:
            for token in doc:
                if token.pos_ in pos_tags:
                    pos_counts[token.pos_] += 1
        
        # Convert to percentages
        total = sum(pos_counts.values())
        pos_percent = {tag: count/total * 100 for tag, count in pos_counts.items()}
        
        tags, percents = zip(*sorted(pos_percent.items(), key=lambda x: x[1], reverse=True))
        sns.barplot(x=list(percents), y=list(tags), ax=ax, palette="viridis")
        ax.set_title(f'POS Distribution: {id2label[cls]}', fontsize=14)
        ax.set_xlabel('Percentage (%)')
        ax.set_xlim(0, 40)
    
    plt.tight_layout()
    return fig

def plot_top_bigrams(train_df, id2label, n=15):
    """Show most common bigrams per class"""
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 2, figure=fig)
    
    classes = sorted(train_df['label'].unique())
    
    for i, cls in enumerate(classes):
        ax = fig.add_subplot(gs[i])
        class_texts = train_df[train_df['label'] == cls]['text_clean']
        
        vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words='english', 
                                    max_features=n, min_df=2)
        X = vectorizer.fit_transform(class_texts)
        
        bigrams = vectorizer.get_feature_names_out()
        frequencies = X.sum(axis=0).A1
        
        # Sort by frequency
        sorted_idx = frequencies.argsort()
        bigrams_sorted = [bigrams[i] for i in sorted_idx]
        freqs_sorted = frequencies[sorted_idx]
        
        sns.barplot(x=freqs_sorted, y=bigrams_sorted, ax=ax, palette="mako")
        ax.set_title(f'Top {n} Bigrams: {id2label[cls]}', fontsize=14)
        ax.set_xlabel('Frequency')
        ax.set_xlim(0, max(freqs_sorted)*1.1)
    
    plt.tight_layout()
    return fig

def plot_sentiment_distribution(train_df, id2label):
    """Analyze sentiment distribution per class"""
    print("Calculating sentiment scores...")
    # Use batch processing for sentiment analysis
    sentiments = []
    for text in tqdm(train_df['text_clean']):
        sentiments.append(TextBlob(text).sentiment.polarity)
    
    train_df['sentiment'] = sentiments
    
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='label', y='sentiment', data=train_df, 
               palette="Set2", showfliers=False, width=0.6)
    plt.title('Sentiment Distribution by Class', fontsize=16)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Sentiment Polarity', fontsize=12)
    plt.xticks(ticks=range(len(id2label)), labels=[id2label[i] for i in range(len(id2label))], 
               rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.axhline(0, color='r', linestyle='--', alpha=0.5)
    return plt.gcf()

# New visualization: Feature correlations heatmap
def plot_feature_correlations(train_df):
    """Visualize correlations between features"""
    features = ['char_count', 'word_count', 'sentence_count', 
               'avg_word_length', 'avg_sentence_length']
    
    # Compute correlations
    corr = train_df[features].corr()
    
    plt.figure(figsize=(12, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", 
               linewidths=0.5, cbar_kws={"shrink": 0.8}, mask=mask)
    plt.title('Feature Correlations', fontsize=16)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    return plt.gcf()

# New visualization: Length vs class
def plot_length_vs_class(train_df, id2label):
    """Visualize text length distribution per class"""
    plt.figure(figsize=(14, 8))
    sns.violinplot(x='label', y='word_count', data=train_df, 
                  palette="Set2", inner="quartile", cut=0)
    plt.title('Word Count Distribution by Class', fontsize=16)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Word Count', fontsize=12)
    plt.xticks(ticks=range(len(id2label)), labels=[id2label[i] for i in range(len(id2label))], 
               rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 2500)  # Reasonable limit
    return plt.gcf()

# Main function to generate all visualizations
def visualize_data(train_path, test_path, output_dir="visualizations"):
    """Generate all visualizations and save to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Label mapping
    id2label = {
        0: "Human", 1: "ChatGPT", 2: "Cohere", 
        3: "Davinci", 4: "Bloomz", 5: "Dolly"
    }
    
    print("Loading data...")
    train_df, test_df = load_data(train_path, test_path)
    
    print("\nPreprocessing train data:")
    train_df = preprocess_data(train_df)
    
    print("\nPreprocessing test data:")
    test_df = preprocess_data(test_df)
    
    print(f"\nTrain samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Generate and save visualizations
    visualizations = [
        # ("class_distribution", plot_class_distribution(train_df, test_df, id2label)),
        # ("text_length", plot_text_length_distribution(train_df, test_df)),
        ("word_count_by_class", plot_length_vs_class(train_df, id2label)),
        # ("pos_distribution", plot_pos_distribution(train_df, id2label)),
        # ("sentiment", plot_sentiment_distribution(train_df, id2label)),
        # ("bigrams", plot_top_bigrams(train_df, id2label)),
        # ("feature_correlations", plot_feature_correlations(train_df)),
        # ("feature_distributions", plot_feature_distributions(train_df, test_df))
    ]
    
    print("\nGenerating visualizations:")
    for name, fig in tqdm(visualizations, desc="Visualizations"):
        fig_path = os.path.join(output_dir, f"{name}.png")
        fig.savefig(fig_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"  Saved {fig_path}")
    
    print("\nAll visualizations saved successfully!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate data visualizations for text classification")
    parser.add_argument("--train", required=True, help="Path to train JSONL file")
    parser.add_argument("--test", required=True, help="Path to test JSONL file")
    parser.add_argument("--output_dir", default="visualizations", help="Output directory for visualizations")
    args = parser.parse_args()
    
    visualize_data(args.train, args.test, args.output_dir)