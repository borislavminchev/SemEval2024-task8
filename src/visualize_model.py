import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, classification_report
from scipy.special import softmax
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import warnings
import argparse
from traditional import extract_stylometric_features
from scipy.sparse import hstack

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Visualization settings
plt.style.use('ggplot')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'DejaVu Sans',
    'axes.facecolor': 'white',
    'figure.facecolor': 'white'
})
sns.set_palette("viridis")

def load_training_history(model_dir):
    """Load training history from TensorBoard logs"""
    log_dir = os.path.join(model_dir, "logs")
    if not os.path.exists(log_dir):
        logging.warning(f"Log directory not found: {log_dir}")
        return None
    
    event_files = [f for f in os.listdir(log_dir) if f.startswith("events.out.tfevents")]
    if not event_files:
        logging.warning("No TensorBoard event files found")
        return None
    
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()
        
        history = {}
        for tag in event_acc.Tags()['scalars']:
            events = event_acc.Scalars(tag)
            history[tag] = [e.value for e in events]
            
        return history
    except ImportError:
        logging.error("TensorBoard not installed. Skipping history loading.")
        return None
    except Exception as e:
        logging.error(f"Error loading TensorBoard logs: {str(e)}")
        return None

def plot_training_curves(history, output_dir):
    """Create learning curve visualizations"""
    if not history:
        logging.warning("No training history available")
        return None
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "training_curves.png")
    
    # Determine available metrics
    available_metrics = set(history.keys())
    metrics_to_plot = {
        'loss': ['train/loss', 'eval/loss'],
        'f1': ['eval/f1'],
        'lr': ['train/learning_rate']
    }
    
    # Create appropriate number of subplots
    num_plots = sum(1 for metrics in metrics_to_plot.values() 
                   if any(m in available_metrics for m in metrics))
    
    if num_plots == 0:
        logging.warning("No valid metrics found for plotting")
        return None
    
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots))
    if num_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Loss plot
    if all(m in available_metrics for m in metrics_to_plot['loss']):
        train_loss = history.get('train/loss', [])
        eval_loss = history.get('eval/loss', [])
        
        if train_loss or eval_loss:
            ax = axes[plot_idx]
            if train_loss:
                ax.plot(train_loss, label='Training Loss')
            if eval_loss:
                ax.plot(eval_loss, label='Validation Loss')
            
            ax.set_title('Training & Validation Loss')
            ax.set_xlabel('Epoch' if len(train_loss) > 1 else 'Step')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1
    
    # F1 Score plot
    if any(m in available_metrics for m in metrics_to_plot['f1']):
        for metric in metrics_to_plot['f1']:
            if metric in history:
                ax = axes[plot_idx]
                ax.plot(history[metric], label='Validation F1', color='green')
                ax.set_title('Validation F1 Score')
                ax.set_xlabel('Epoch' if len(history[metric]) > 1 else 'Step')
                ax.set_ylabel('F1 Score')
                ax.set_ylim(0, 1)
                ax.legend()
                ax.grid(True, alpha=0.3)
                plot_idx += 1
                break
    
    # Learning rate plot
    if any(m in available_metrics for m in metrics_to_plot['lr']):
        for metric in metrics_to_plot['lr']:
            if metric in history:
                ax = axes[plot_idx]
                ax.plot(history[metric], label='Learning Rate', color='purple')
                ax.set_title('Learning Rate Schedule')
                ax.set_xlabel('Step')
                ax.set_ylabel('LR')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plot_idx += 1
                break
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved training curves to {output_path}")
    return fig

def plot_confusion_matrix(y_true, y_pred, labels, title, output_path):
    """Create confusion matrix visualization"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    
    # Absolute counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0], 
                xticklabels=labels, yticklabels=labels)
    ax[0].set_title(f'{title} - Absolute Counts')
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('True')
    
    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax[1],
                xticklabels=labels, yticklabels=labels)
    ax[1].set_title(f'{title} - Normalized')
    ax[1].set_xlabel('Predicted')
    ax[1].set_ylabel('True')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved confusion matrix to {output_path}")
    return fig

def plot_classification_report(y_true, y_pred, labels, title, output_path):
    """Visualize classification report as heatmap"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    report_df = pd.DataFrame(report).iloc[:-1, :].T
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(report_df, annot=True, cmap="YlGnBu", fmt='.2f',
                cbar_kws={"shrink": 0.8}, vmin=0, vmax=1)
    plt.title(f'Classification Report: {title}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved classification report to {output_path}")
    return plt.gcf()

def plot_roc_curves(y_true, y_probs, class_names, output_path):
    """Plot ROC curves for multi-class classification"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    
    # Compute ROC curve and ROC area for each class
    fpr, tpr, roc_auc = {}, {}, {}
    plt.figure(figsize=(10, 8))
    
    for i, name in enumerate(class_names):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2, 
                 label=f'Class {name} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved ROC curves to {output_path}")
    return plt.gcf()

def plot_probability_distribution(y_probs, class_names, output_path):
    """Visualize prediction probability distributions"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    n_classes = len(class_names)
    n_cols = min(3, n_classes)
    n_rows = (n_classes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_classes > 1 else [axes]
    
    for i, name in enumerate(class_names):
        ax = axes[i]
        sns.histplot(y_probs[:, i], bins=20, kde=True, ax=ax, color='skyblue')
        ax.set_title(f'Probability Distribution: {name}', fontsize=12)
        ax.set_xlabel('Probability')
        ax.set_ylabel('Frequency')
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved probability distribution to {output_path}")
    return fig

def plot_confidence_vs_accuracy(y_true, y_pred, y_probs, output_path):
    """Analyze relationship between confidence and accuracy"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    correct = (y_true == y_pred).astype(int)
    max_probs = np.max(y_probs, axis=1)
    
    # Bin probabilities
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    accuracy_per_bin = []
    counts_per_bin = []
    confidence_per_bin = []
    
    for i in range(len(bins) - 1):
        mask = (max_probs >= bins[i]) & (max_probs < bins[i+1])
        if mask.sum() > 0:
            acc = correct[mask].mean()
            conf = max_probs[mask].mean()
            accuracy_per_bin.append(acc)
            counts_per_bin.append(mask.sum())
            confidence_per_bin.append(conf)
        else:
            accuracy_per_bin.append(0)
            counts_per_bin.append(0)
            confidence_per_bin.append(0)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Accuracy and confidence
    ax1.plot(bin_centers, accuracy_per_bin, 'o-', color='royalblue', label='Accuracy')
    ax1.plot(bin_centers, confidence_per_bin, 'o-', color='darkorange', label='Confidence')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
    ax1.set_xlabel('Confidence Bin')
    ax1.set_ylabel('Accuracy / Confidence')
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)
    
    # Sample count
    ax2 = ax1.twinx()
    ax2.bar(bin_centers, counts_per_bin, width=0.08, alpha=0.3, color='green', label='Samples')
    ax2.set_ylabel('Sample Count')
    ax2.legend(loc='upper right')
    
    plt.title('Confidence vs Accuracy', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved confidence vs accuracy plot to {output_path}")
    return plt.gcf()

def plot_calibration_curve(y_true, y_probs, class_names, output_path):
    """Plot calibration curves for each class"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    n_classes = len(class_names)
    n_cols = min(3, n_classes)
    n_rows = (n_classes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_classes > 1 else [axes]
    
    for i, name in enumerate(class_names):
        ax = axes[i]
        prob_true, prob_pred = calibration_curve(
            (y_true == i), y_probs[:, i], n_bins=10, strategy='quantile'
        )
        
        ax.plot(prob_pred, prob_true, marker='o', linewidth=1, label=name)
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect')
        ax.set_title(f'Calibration: {name}')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved calibration curves to {output_path}")
    return fig

def plot_error_analysis(y_true, y_pred, texts, class_names, output_path):
    """Visualize error patterns in misclassifications"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    errors = []
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true != pred:
            errors.append({
                'text': texts[i],
                'true_class': class_names[true],
                'predicted_class': class_names[pred]
            })
    
    if not errors:
        logging.warning("No errors found for analysis")
        return None
    
    error_df = pd.DataFrame(errors)
    
    # Plot error distribution
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    # Error by true class
    sns.countplot(data=error_df, x='true_class', ax=ax[0], order=class_names)
    ax[0].set_title('Misclassified by True Class')
    ax[0].set_xlabel('True Class')
    ax[0].set_ylabel('Count')
    ax[0].tick_params(axis='x', rotation=45)
    
    # Error by predicted class
    sns.countplot(data=error_df, x='predicted_class', ax=ax[1], order=class_names)
    ax[1].set_title('Misclassified by Predicted Class')
    ax[1].set_xlabel('Predicted Class')
    ax[1].set_ylabel('Count')
    ax[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved error analysis to {output_path}")
    return fig

def visualize_model_performance(model_type, model_path, predictions_path, output_dir, class_names, gold_file):
    """Main function to generate all performance visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Starting visualization for {model_type} model")
    
    # Load predictions and gold data
    try:
        preds_df = pd.read_json(predictions_path, lines=True)
        gold_df = pd.read_json(gold_file, lines=True)
        
        # Merge predictions with gold labels
        if 'id' in gold_df.columns and 'id' in preds_df.columns:
            merged_df = preds_df.merge(gold_df[['id', 'label', 'text']], on='id', suffixes=('', '_true'))
            merged_df = merged_df.rename(columns={'label': 'predicted', 'label_true': 'true'})
        else:
            logging.error("ID column missing in gold or predictions data")
            return
            
        # Prepare data arrays
        y_true = merged_df['true'].values
        y_pred = merged_df['predicted'].values
        texts = merged_df['text'].values
        
        # Handle probabilities if available
        y_probs = None
        if model_type == "traditional" and 'probabilities' not in merged_df.columns:
            print("Computing probabilities for traditional model...")
            try:
                # Load model artifacts
                model = joblib.load(model_path + ".pkl")
                tfidf = joblib.load(model_path + "_tfidf.pkl")
                scaler = joblib.load(model_path + "_scaler.pkl")
                
                selector_path = model_path + "_selector.pkl"
                if os.path.exists(selector_path):
                    selector = joblib.load(selector_path)
                    logging.info("Loaded feature selector")
                else:
                    selector = None
                    logging.info("No feature selector found")
                # Extract features from text
                texts = merged_df['text'].tolist()
                tfidf_features = tfidf.transform(texts)
                stylo_features = extract_stylometric_features(texts)
                stylo_features = scaler.transform(stylo_features)
                features = hstack([tfidf_features, stylo_features])

                 # Apply feature selection if available
                if selector:
                    features = selector.transform(features)
                    logging.info(f"Applied feature selection. New shape: {features.shape}")
                
                # Compute probabilities
                y_probs = model.predict_proba(features)
                merged_df['probabilities'] = y_probs.tolist()
                print("Successfully computed probabilities")
            except Exception as e:
                print(f"Couldn't compute probabilities: {str(e)}")
        elif 'probabilities' in merged_df.columns:
            try:
                # Convert from string if needed
                if isinstance(merged_df['probabilities'].iloc[0], str):
                    merged_df['probabilities'] = merged_df['probabilities'].apply(eval)
                
                # Convert to numpy array
                y_probs = np.stack(merged_df['probabilities'].values)
            except Exception as e:
                logging.error(f"Error processing probabilities: {str(e)}")
                y_probs = None
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        return
    
    if y_probs is None and model_type == "neural":
        try:
            logging.info("Computing probabilities from model...")
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = AutoTokenizer.from_pretrained(model_path+'best')
            model = AutoModelForSequenceClassification.from_pretrained(model_path+'best').to(device)
            
            # Ensure all texts are strings
            texts = [str(t) if t is not None else "" for t in texts]
            
            # Batch processing
            batch_size = 32
            y_probs = []
            for i in range(0, len(texts), batch_size):
                inputs = tokenizer(
                    texts[i:i+batch_size],
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
                ).to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    batch_probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                    y_probs.append(batch_probs)
                    
            y_probs = np.vstack(y_probs)
        except Exception as e:
            logging.error(f"Probability computation failed: {str(e)}")
            y_probs = None
    
        # Generate core visualizations
    plot_confusion_matrix(
        y_true, y_pred, class_names, 
        f"{model_type.capitalize()} Model",
        os.path.join(output_dir, "confusion_matrix.png")
    )
    
    plot_classification_report(
        y_true, y_pred, class_names, 
        f"{model_type.capitalize()} Model",
        os.path.join(output_dir, "classification_report.png")
    )
    
    # Only attempt probability-based visualizations if y_probs is valid
    if y_probs is not None and isinstance(y_probs, np.ndarray):
        plot_probability_distribution(
            y_probs, class_names, 
            os.path.join(output_dir, "probability_distribution.png")
        )
        
        plot_confidence_vs_accuracy(
            y_true, y_pred, y_probs,
            os.path.join(output_dir, "confidence_vs_accuracy.png")
        )
        
        plot_calibration_curve(
            y_true, y_probs, class_names,
            os.path.join(output_dir, "calibration_curve.png")
        )
        
        plot_roc_curves(
            y_true, y_probs, class_names,
            os.path.join(output_dir, "roc_curves.png")
        )
    else:
        logging.warning("Skipping probability-based visualizations - no valid probability data")
    
    
    # Error analysis
    plot_error_analysis(
        y_true, y_pred, texts, class_names,
        os.path.join(output_dir, "error_analysis.png")
    )
    
    # Model-specific visualizations
    if model_type == "neural":
        history = load_training_history(model_path)
        if history:
            plot_training_curves(history, output_dir)
    
    logging.info(f"All visualizations saved to: {output_dir}")

def compare_models(model_results, output_dir):
    """Compare multiple models across key metrics"""
    os.makedirs(output_dir, exist_ok=True)
    
    if not model_results:
        logging.error("No model results provided for comparison")
        return
    
    # Create metrics comparison
    metrics_df = pd.DataFrame(model_results).T
    plt.figure(figsize=(12, 8))
    metrics_df.plot(kind='bar', rot=0)
    plt.title('Model Performance Comparison', fontsize=16)
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "metric_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved metric comparison to {output_path}")
    
    # Create interactive comparison
    try:
        fig = make_subplots(rows=1, cols=len(metrics_df.columns), 
                           subplot_titles=metrics_df.columns,
                           shared_yaxes=True)
        
        for i, metric in enumerate(metrics_df.columns, start=1):
            fig.add_trace(
                go.Bar(x=metrics_df.index, y=metrics_df[metric], name=metric),
                row=1, col=i
            )
        
        fig.update_layout(
            title_text="Model Performance Comparison",
            height=500,
            showlegend=False,
            template="plotly_white"
        )
        
        output_path = os.path.join(output_dir, "interactive_comparison.html")
        fig.write_html(output_path)
        logging.info(f"Saved interactive comparison to {output_path}")
    except ImportError:
        logging.warning("Plotly not installed. Skipping interactive visualization.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Performance Visualization Toolkit")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Visualize single model
    model_parser = subparsers.add_parser('visualize', help='Visualize single model performance')
    model_parser.add_argument('--model_type', required=True, choices=['neural', 'traditional'],
                             help='Type of model')
    model_parser.add_argument('--model_path', required=True,
                             help='Path to model directory or file')
    model_parser.add_argument('--predictions', required=True,
                             help='Path to predictions JSONL file')
    model_parser.add_argument('--output_dir', required=True,
                             help='Output directory for visualizations')
    model_parser.add_argument('--class_names', nargs='+', required=True,
                             help='List of class names in order')
    model_parser.add_argument('--gold_file', required=True,
                             help='Path to gold standard JSONL file with true labels')
    
    # Compare multiple models
    compare_parser = subparsers.add_parser('compare', help='Compare multiple models')
    compare_parser.add_argument('--results', required=True,
                               help='JSON file with model results')
    compare_parser.add_argument('--output_dir', required=True,
                               help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    if args.command == 'visualize':
        visualize_model_performance(
            model_type=args.model_type,
            model_path=args.model_path,
            predictions_path=args.predictions,
            output_dir=args.output_dir,
            class_names=args.class_names,
            gold_file=args.gold_file
        )
    elif args.command == 'compare':
        try:
            with open(args.results, 'r') as f:
                model_results = json.load(f)
            compare_models(model_results, args.output_dir)
        except Exception as e:
            logging.error(f"Error loading comparison results: {str(e)}")
    else:
        parser.print_help()


# python .\src\visualize_model.py visualize --model_type neural --model_path .\checkpoints\distilroberta-base_subtaskB\ --predictions .\subtaskB_predictions_123.jsonl --output_dir neural_viz --class_names human chatGPT cohere davinci bloomz dolly --gold_file .\gold_test\subtaskB.jsonl
# python .\src\visualize_model.py visualize --model_type traditional --model_path ./ml_models/rf --predictions ./ml_models/preds_rf.jsonl --output_dir traditional_viz --class_names human chatGPT cohere davinci bloomz dolly --gold_file .\B_balanced_test.jsonl