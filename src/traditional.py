import os
import argparse
import logging

import pandas as pd
import numpy as np
import joblib
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import spacy.symbols
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectFromModel

import re
import emoji
import textstat

# ─── Set up ────────────────────────────────────────────────────────────────────

nlp = spacy.load("en_core_web_sm", disable=["parser"])
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

def clean_text(text):
    # 1) Normalize case
    text = text.lower()
    # 2) Replace URLs
    text = re.sub(r'https?://\S+', ' <URL> ', text)
    # 3) Strip HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # 4) Strip mentions/hashtags
    text = re.sub(r'@\w+|#\w+', ' ', text)
    # 5) Convert emojis to text (optional)
    text = emoji.demojize(text, delimiters=(" ", " "))
    # 6) Remove unwanted characters (keep basic punctuation)
    text = re.sub(r'[^a-z0-9<>\[\]_.,!?\s]', ' ', text)
    # 7) Collapse whitespace
    text = ' '.join(text.split()).strip()
    # 8) Handle blanks
    return text if text else "[BLANK]"

# ─── Data ─────────────────────────────────────────────────────────────────────

def get_data(train_path, test_path, random_seed):
    # Load raw data
    train_df = pd.read_json(train_path, lines=True)
    test_df  = pd.read_json(test_path,  lines=True)

    # Apply cleaning to text column
    train_df['text'] = train_df['text'].map(clean_text)
    test_df['text']  = test_df['text'].map(clean_text)

    # Split into train/validation with stratification
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        stratify=train_df['label'],
        random_state=random_seed
    )
    return train_df, val_df, test_df


# ─── Features ─────────────────────────────────────────────────────────────────

def extract_stylometric_features(texts):
    # Add new feature names for clarity, though not used in the code itself
    # feature_names = [
    #     "num_chars", "num_words", "avg_word_len", "avg_sent_len",
    #     "punct_count", "ttr", "noun_ratio", "verb_ratio", "adj_ratio",
    #     "flesch_reading_ease", "flesch_kincaid_grade", "gunning_fog"
    # ]
    feats = []
    for doc in nlp.pipe(texts, batch_size=64, n_process=-1):
        # --- Existing Features ---
        num_chars = len(doc.text)
        words = [t for t in doc if not t.is_punct and not t.is_space]
        num_words = len(words)
        avg_word_len = np.mean([len(t.text) for t in words]) if words else 0
        num_sents = max(1, len(list(doc.sents)))
        avg_sent_len = num_words / num_sents

        punct_count = sum(1 for t in doc if t.is_punct)
        alpha_tokens = [t for t in doc if t.is_alpha]
        num_alpha = len(alpha_tokens)
        ttr = len({t.text.lower() for t in alpha_tokens}) / max(1, num_alpha)

        pos_counts = doc.count_by(spacy.symbols.POS)
        # noun_ratio = pos_counts.get(nlp.vocab.strings["NOUN"], 0) / max(1, num_words)
        # verb_ratio = pos_counts.get(nlp.vocab.strings["VERB"], 0) / max(1, num_words)
        # adj_ratio = pos_counts.get(nlp.vocab.strings["ADJ"], 0) / max(1, num_words)
        noun_ratio = sum(1 for t in doc if t.pos_ == "NOUN") / max(1, num_words)
        verb_ratio = sum(1 for t in doc if t.pos_ == "VERB") / max(1, num_words)
        adj_ratio = sum(1 for t in doc if t.pos_ == "ADJ") / max(1, num_words)

        # --- New Readability Features ---
        # The textstat library works on the raw text string
        raw_text = doc.text
        flesch_reading_ease = textstat.flesch_reading_ease(raw_text)
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(raw_text)
        gunning_fog = textstat.gunning_fog(raw_text)
        coleman_liau = textstat.coleman_liau_index(raw_text)
        automated_readability = textstat.automated_readability_index(raw_text)
        # --- End of New Features ---

        feats.append([
            num_chars, num_words, avg_word_len, avg_sent_len,
            punct_count, ttr, noun_ratio, verb_ratio, adj_ratio,
            # --- Append new features to the list ---
            flesch_reading_ease, flesch_kincaid_grade, gunning_fog,
            coleman_liau, automated_readability
        ])
    return np.array(feats)




def extract_features(train_texts, val_texts, test_texts, max_tfidf_features=5000):
    # 1) TF-IDF
    tfidf = TfidfVectorizer(
        max_features=max_tfidf_features,
        ngram_range=(1, 2),
        stop_words='english'
    )
    X_tr_tfidf = tfidf.fit_transform(train_texts)
    X_va_tfidf = tfidf.transform(val_texts)
    X_te_tfidf = tfidf.transform(test_texts)

    # 2) Stylometric
    X_tr_sty = extract_stylometric_features(train_texts)
    X_va_sty = extract_stylometric_features(val_texts)
    X_te_sty = extract_stylometric_features(test_texts)

    scaler = StandardScaler()
    X_tr_sty = scaler.fit_transform(X_tr_sty)
    X_va_sty = scaler.transform(X_va_sty)
    X_te_sty = scaler.transform(X_te_sty)

    # 3) Combine
    X_train = hstack([X_tr_tfidf, X_tr_sty])
    X_valid = hstack([X_va_tfidf, X_va_sty])
    X_test  = hstack([X_te_tfidf, X_te_sty])

    return X_train, X_valid, X_test, tfidf, scaler


# ─── Models ───────────────────────────────────────────────────────────────────

def train_model(X_train, y_train, model_type="rf", feature_selection=True):
    if model_type == "rf":
        base_model = RandomForestClassifier(
            class_weight='balanced',
            n_estimators=100,  # Smaller for feature selection
            max_depth=10,
            n_jobs=-1,
            random_state=0
        )
    elif model_type == "xgb":
        if isinstance(X_train, np.ndarray):
            X_train = csr_matrix(X_train)
        base_model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=0
        )

    # Feature selection using model-based importance
    if feature_selection:
        selector = SelectFromModel(
            estimator=base_model,
            threshold="1.25*median",  # Tune this
            max_features=1500  # Keep top features
        )
        initial_size = X_train.shape[1]
        X_train = selector.fit_transform(X_train, y_train)
        logging.info(f"Selected {X_train.shape[1]} features out of {initial_size}")
    else:
        selector = None

    
    if model_type == "rf":
        model = RandomForestClassifier(
            class_weight='balanced',
            n_estimators=200,
            max_depth=15,
            n_jobs=-1,
            random_state=0
        )
    elif model_type == "xgb":
        # Convert to CSR format for XGBoost efficiency
        if isinstance(X_train, np.ndarray):
            X_train = csr_matrix(X_train)
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            tree_method='hist',  # Optimized for sparse data
            random_state=0
        )
    
    model.fit(X_train, y_train)
    return model, selector  # Return selector for test set


def evaluate_model(model, X, y, target_names=None):
    preds = model.predict(X)
    report = classification_report(y, preds, target_names=target_names, zero_division=0)
    return report, preds


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train RF/XGB on text with enriched features")
    p.add_argument("--train",  "-tr", required=True, help="Train JSONL file path")
    p.add_argument("--test",   "-t",  required=True, help="Test  JSONL file path")
    p.add_argument("--model",  "-m",  choices=["rf","xgb"], default="rf", help="Which model to run")
    p.add_argument("--seed",   "-s",  type=int, default=0, help="Random seed")
    p.add_argument("--outdir", "-o",  default="./ml_out", help="Where to save models & preds")
    p.add_argument("--feature_sel", action="store_true", help="Enable feature selection")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    logging.info(f"Loading data with seed={args.seed}")
    train_df, valid_df, test_df = get_data(args.train, args.test, random_seed=args.seed)

    logging.info("Extracting features")
    X_train, X_val, X_test, tfidf, scaler = extract_features(
        train_df["text"], valid_df["text"], test_df["text"]
    )
    y_train = train_df["label"]
    y_val   = valid_df["label"]
    y_test  = test_df["label"]

    logging.info(f"Training {args.model.upper()}")
    model, selector = train_model(X_train, y_train, 
                                 model_type=args.model,
                                 feature_selection=args.feature_sel)
    
    if selector:
        X_val = selector.transform(X_val)
        X_test = selector.transform(X_test)

    logging.info("Evaluating on validation set")
    val_report, _ = evaluate_model(model, X_val, y_val)
    print("Validation Classification Report:\n", val_report)

    logging.info("Evaluating on test set")
    test_report, test_preds = evaluate_model(
        model, X_test, y_test,
        target_names=[str(l) for l in sorted(set(y_train))]
    )
    print("Test Classification Report:\n", test_report)

    # ─── Save artifacts ────────────────────────────────────────────────────
    base = os.path.join(args.outdir, args.model)
    logging.info(f"Saving model → {base}.pkl")
    joblib.dump(model, base + ".pkl")
    joblib.dump(tfidf, base + "_tfidf.pkl")
    joblib.dump(scaler, base + "_scaler.pkl")

    if selector:
        joblib.dump(selector, base + "_selector.pkl")

    pred_df = pd.DataFrame({
        "id":    test_df["id"],
        "label": test_preds
    })
    pred_path = os.path.join(args.outdir, f"preds_{args.model}.jsonl")
    logging.info(f"Saving predictions → {pred_path}")
    pred_df.to_json(pred_path, orient="records", lines=True)

    logging.info("All done!")

# python .\subtaskA\baseline\app.py --train ./B_balanced_train.jsonl --test ./B_balanced_test.jsonl --model rf --outdir ./ml_models --seed 42                                                                                                 
