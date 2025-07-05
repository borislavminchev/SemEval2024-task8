import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import spacy
import matplotlib.pyplot as plt
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import re
import emoji
from spacy import symbols
import textstat

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="AI Text Detector", 
    page_icon=":robot_face:", 
    layout="wide"
)

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

# ──── Configuration ────────────────────────────────────────────────────────────
id2label = {
    0: "Human", 1: "ChatGPT", 2: "Cohere", 
    3: "Davinci", 4: "Bloomz", 5: "Dolly"
}
class_colors = {
    "Human": "#4C72B0", "ChatGPT": "#DD8452", "Cohere": "#55A868",
    "Davinci": "#C44E52", "Bloomz": "#8172B3", "Dolly": "#937860"
}

# ──── Model Paths ─────────────────────────────────────────────────────────────
NEURAL_MODEL_PATH =  "./checkpoints/xlm-roberta-base_custom_classifier/checkpoint-7548" #"./checkpoints/distilroberta-base_subtaskB/best"

TRADITIONAL_MODEL_PATHS = {
    "rf": {
        "model": "./ml_models/rf.pkl",
        "tfidf": "./ml_models/rf_tfidf.pkl",
        "scaler": "./ml_models/rf_scaler.pkl",
        "selector": "./ml_models/rf_selector.pkl"
    },
    "xgb": {
        "model": "./ml_models/xgb.pkl",
        "tfidf": "./ml_models/xgb_tfidf.pkl",
        "scaler": "./ml_models/xgb_scaler.pkl",
        "selector": "./ml_models/xgb_selector.pkl"
    }
}

# ──── Text Cleaning ────────────────────────────────────────────────────────────
def clean_text(text, lowercase=True):
    """Unified text cleaning function"""

    if lowercase:
        text = text.lower()
    text = re.sub(r'https?://\S+', ' <URL> ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'@\w+|#\w+', ' ', text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r'[^a-z0-9<>\[\]_.,!?\s]', ' ', text)
    text = " ".join(text.split()).strip()
    return text if text else "[BLANK]"

# ──── SpaCy Setup ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_spacy_model():
    """Load and cache SpaCy model"""
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp

nlp = load_spacy_model()

# ──── Feature Extraction (FIXED to match training) ────────────────────────────
# def extract_stylometric_features(texts):
#     """Extract 9 stylometric features to match training"""
#     feats = []
#     for doc in nlp.pipe(texts, batch_size=64):
#         num_chars = len(doc.text)
#         num_words = len(doc)
#         tokens_for_word_len = [t for t in doc if not t.is_punct and not t.is_space]
#         avg_word_len = np.mean([len(t.text) for t in tokens_for_word_len]) if tokens_for_word_len else 0
#         num_sents = max(1, len(list(doc.sents)))
#         avg_sent_len = num_words / num_sents

#         punct_count = sum(1 for t in doc if t.is_punct)
#         ttr = len({t.text.lower() for t in doc if t.is_alpha}) / max(1, num_words)

#         # POS features
#         pos_counts = doc.count_by(symbols.POS)
#         noun_ratio = pos_counts.get(nlp.vocab.strings["NOUN"], 0) / max(1, num_words)
#         verb_ratio = pos_counts.get(nlp.vocab.strings["VERB"], 0) / max(1, num_words)
#         adj_ratio = pos_counts.get(nlp.vocab.strings["ADJ"], 0) / max(1, num_words)

#         feats.append([
#             num_chars, num_words, avg_word_len, avg_sent_len,
#             punct_count, ttr, noun_ratio, verb_ratio, adj_ratio
#         ])
#     return np.array(feats)

# ──── Model Loading ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=64)
def load_neural_model():
    """Load neural model and tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(NEURAL_MODEL_PATH)
        if tokenizer.pad_token is None: # gpt2
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model = AutoModelForSequenceClassification.from_pretrained(NEURAL_MODEL_PATH)
        model.resize_token_embeddings(len(tokenizer)) # gpt2
        model.config.pad_token_id = tokenizer.pad_token_id # gpt2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Failed to load neural model: {str(e)}")
        st.stop()

@st.cache_data(show_spinner=False, max_entries=64)
def load_traditional_model(model_type="rf"):
    """Load traditional model and feature processors"""
    try:
        paths = TRADITIONAL_MODEL_PATHS[model_type]
        model = joblib.load(paths["model"])
        tfidf = joblib.load(paths["tfidf"])
        scaler = joblib.load(paths["scaler"])
        selector = joblib.load(paths["selector"])
        return model, tfidf, scaler, selector
    except Exception as e:
        st.error(f"Failed to load traditional model: {str(e)}")
        st.stop()

# ──── Prediction Functions ────────────────────────────────────────────────────
def neural_predict(texts, tokenizer, model, device, max_length=256):
    """Predict using neural model"""
    cleaned_texts = [clean_text(t, lowercase=False) for t in texts]
    inputs = tokenizer(
        cleaned_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits.cpu().numpy()
    probs = softmax(logits, axis=1)
    return probs

SENT_END_RE = re.compile(r'[.!?]+')
WORD_RE     = re.compile(r'\w+')
import string

def extract_stylometric_features(texts: list[str]) -> np.ndarray:
    feats = []
    for text in texts:
        # words = sequences of alphanumeric characters
        words = WORD_RE.findall(text)
        num_words = len(words)
        num_chars = len(text)
        avg_word_len = np.mean([len(w) for w in words]) if words else 0.0

        # sentences ≈ chunks split on ., ! or ?
        sents = [s for s in SENT_END_RE.split(text) if s.strip()]
        num_sents = max(1, len(sents))
        avg_sent_len = num_words / num_sents if num_words else 0.0

        # punctuation count
        punct_count = sum(1 for c in text if c in string.punctuation)

        # type‐token ratio
        ttr = len({w.lower() for w in words}) / num_words if num_words else 0.0

        # POS proxies: naive heuristics based on word-endings
        doc = nlp(text)
        noun_ratio = sum(1 for t in doc if t.pos_ == "NOUN") / max(1, num_words)
        print("check12")
        verb_ratio = sum(1 for t in doc if t.pos_ == "VERB") / max(1, num_words)
        print("check13")
        adj_ratio = sum(1 for t in doc if t.pos_ == "ADJ") / max(1, num_words)

        flesch_reading_ease = textstat.flesch_reading_ease(text)
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
        gunning_fog = textstat.gunning_fog(text)
        coleman_liau = textstat.coleman_liau_index(text)
        automated_readability = textstat.automated_readability_index(text)

        feats.append([
            num_chars, num_words, avg_word_len, avg_sent_len,
            punct_count, ttr, noun_ratio, verb_ratio, adj_ratio,
            flesch_reading_ease, flesch_kincaid_grade, gunning_fog,
            coleman_liau, automated_readability
        ])
    return np.array(feats)

@st.cache_data(show_spinner=False, max_entries=64)
def traditional_predict(texts, model_type="rf"):
    # load models (this is fast thanks to cache_resource)
    model, tfidf, scaler, selector = load_traditional_model(model_type)
    
    # clean & vectorize
    cleaned_texts = [clean_text(t) for t in texts]
    print("da2")
    X_tfidf = tfidf.transform(cleaned_texts)
    print("da3")
    
    # stylometrics
    X_sty = extract_stylometric_features(cleaned_texts)
    
    X_sty = scaler.transform(X_sty)
    
    # combine & select
    X = hstack([X_tfidf, X_sty])
    X = selector.transform(X)
    
    # predict
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    else:
        decisions = model.decision_function(X)
        return softmax(decisions, axis=1)


# ──── Visualization ───────────────────────────────────────────────────────────
def plot_class_probabilities(probs, title):
    """Plot class probabilities as a bar chart"""
    class_names = [id2label[i] for i in range(6)]
    colors = [class_colors[name] for name in class_names]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(class_names, probs, color=colors)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add probability labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.02, 
            f'{height:.2f}',
            ha='center', 
            va='bottom',
            fontsize=10
        )
    
    st.pyplot(fig)


# ──── UI Layout ────────────────────────────────────────────────────────────────
st.title("🧠 AI Text Detector")
st.markdown("Classify text origin using neural networks or traditional ML models")

# Sidebar controls
with st.sidebar:
    st.header("Model Selection")
    model_type = st.radio(
        "Choose Model Type:",
        ("Neural Network", "Traditional ML")
    )
    
    if model_type == "Traditional ML":
        algorithm = st.radio(
            "Algorithm:",
            ("Random Forest", "XGBoost")
        )
        model_type_short = "rf" if algorithm == "Random Forest" else "xgb"
    
    st.header("Text Input")
    input_option = st.radio(
        "Input Method:",
        ("Single Text", "Multiple Texts")
    )
    
    if input_option == "Single Text":
        text_input = st.text_area(
            "Enter text to classify:", 
            height=200,
            placeholder="Paste text here..."
        )
        texts = [text_input] if text_input else []
    else:
        uploaded_file = st.file_uploader(
            "Upload text file (one per line)", 
            type=["txt"]
        )
        if uploaded_file:
            texts = uploaded_file.read().decode("utf-8").splitlines()
        else:
            texts = []
    
    classify_btn = st.button("Classify Text", disabled=not texts)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Classification Results")
    
    if classify_btn and texts:
        with st.spinner("Analyzing text..."):
            try:
                # Load models dynamically based on selection
                if model_type == "Neural Network":
                    tokenizer, model, device = load_neural_model()
                    probs = neural_predict(texts, tokenizer, model, device)
                else:
                    # model, tfidf, scaler, selector = load_traditional_model(model_type_short)
                    print("da")
                    probs = traditional_predict(texts, model_type_short)
                
                # Process results
                for i, text in enumerate(texts):
                    pred_class_idx = np.argmax(probs[i])
                    pred_class = id2label[pred_class_idx]
                    pred_prob = probs[i][pred_class_idx]
                    
                    # Save to history
                    st.session_state.predictions.append({
                        "text": text[:100] + "..." if len(text) > 100 else text,
                        "predicted_class": pred_class,
                        "probability": pred_prob,
                        "model": model_type,
                        "algorithm": algorithm if model_type == "Traditional ML" else "N/A",
                        "timestamp": pd.Timestamp.now()
                    })
                    
                    st.markdown(f"**Predicted Source**: :{pred_class.lower()}: **{pred_class}** ({(pred_prob*100):.1f}%)")
                    plot_class_probabilities(probs[i], f"Classification Probabilities - Text {i+1}")
                    st.markdown("**Full Text:**")
                    st.text(text)
                    
                
                st.success("Classification complete!")
                
            except Exception as e:
                st.error(f"Classification error: {str(e)}")

    elif not texts:
        st.info("Enter text in the sidebar to get started")
    else:
        st.info("Click 'Classify Text' to analyze")

with col2:
    if st.session_state.predictions:
        st.subheader("Analysis Dashboard")
        
        # Latest prediction summary
        latest = st.session_state.predictions[-1]
        st.metric("Latest Prediction", 
                  f"{latest['predicted_class']} ({(latest['probability']*100):.1f}%)")
        
        # Confidence distribution
        st.subheader("Prediction Confidence Distribution")
        conf_df = pd.DataFrame(st.session_state.predictions)
        st.bar_chart(conf_df.groupby("predicted_class")["probability"].mean())
        
    else:
        st.subheader("Analysis Dashboard")
        st.info("Perform classifications to see analytics")
        st.image("https://images.unsplash.com/photo-1677442135722-5fbfd44e05d3?auto=format&fit=crop&w=800", 
                 caption="Text Classification Analysis")

# Footer
st.markdown("---")
st.caption("AI Text Detector v1.0 | Neural models: Transformers | Traditional: TF-IDF + Stylometrics")
