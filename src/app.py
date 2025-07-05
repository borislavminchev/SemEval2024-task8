import os
import json
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


def load_trainer_state(checkpoint_dir):
    state_file = os.path.join(checkpoint_dir, 'trainer_state.json')
    if not os.path.exists(state_file):
        raise FileNotFoundError(f"trainer_state.json not found in {checkpoint_dir}")
    with open(state_file, 'r') as f:
        state = json.load(f)
    history = state.get('log_history', [])
    return pd.DataFrame(history)


def plot_loss_by_epoch(df, output_dir):
    # Aggregate training loss by epoch
    train_logs = df[df['loss'].notnull() & df['epoch'].notnull()]
    train_by_epoch = train_logs.groupby('epoch')['loss'].mean().reset_index()
    # Evaluation loss by epoch
    eval_logs = df[df['eval_loss'].notnull()]
    eval_by_epoch = eval_logs[['epoch', 'eval_loss']].drop_duplicates()

    plt.figure()
    plt.plot(train_by_epoch['epoch'], train_by_epoch['loss'], label='Train Loss')
    plt.plot(eval_by_epoch['epoch'], eval_by_epoch['eval_loss'], label='Eval Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Eval Loss by Epoch')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_by_epoch.png'))
    plt.close()


def plot_eval_metrics(df, output_dir):
    # Eval accuracy and F1 by epoch
    logs = df[df['epoch'].notnull()]
    eval_acc = logs[logs['eval_accuracy'].notnull()][['epoch', 'eval_accuracy']].drop_duplicates()
    eval_f1 = logs[logs['eval_f1'].notnull()][['epoch', 'eval_f1']].drop_duplicates()

    plt.figure()
    plt.plot(eval_acc['epoch'], eval_acc['eval_accuracy'], label='Eval Accuracy')
    plt.plot(eval_f1['epoch'], eval_f1['eval_f1'], label='Eval F1')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Evaluation Accuracy and F1 by Epoch')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'eval_metrics_by_epoch.png'))
    plt.close()


def plot_learning_rate(df, output_dir):
    # Plot learning rate schedule if available
    if 'learning_rate' not in df.columns:
        return
    lr_logs = df[df['learning_rate'].notnull()][['step', 'learning_rate']]
    plt.figure()
    plt.plot(lr_logs['step'], lr_logs['learning_rate'])
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.savefig(os.path.join(output_dir, 'learning_rate_schedule.png'))
    plt.close()


def plot_classification_report(test_file, pred_file, id2label, output_dir):
    # Load true labels and predictions
    test_df = pd.read_json(test_file, lines=True)
    preds_df = pd.read_json(pred_file, lines=True)
    merged = test_df.merge(preds_df, on='id', suffixes=('_true', '_pred'))
    y_true = merged['label_true']
    y_pred = merged['label_pred']
    labels = [id2label[i] for i in sorted(id2label.keys())]

    # Compute report
    report_dict = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    # Drop non-class keys if present
    drop_keys = [k for k in ['accuracy', 'macro avg', 'weighted avg'] if k in report_dict]
    for k in drop_keys:
        report_dict.pop(k)
    report_df = pd.DataFrame(report_dict).T

    # Plot precision, recall, f1-score per class
    plt.figure()
    report_df[['precision', 'recall', 'f1-score']].plot(kind='bar')
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Classification Report Metrics per Class')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classification_report.png'))
    plt.close()


def plot_confusion(test_file, pred_file, id2label, output_dir):
    # Load true labels and predictions
    test_df = pd.read_json(test_file, lines=True)
    preds_df = pd.read_json(pred_file, lines=True)
    if 'label' not in preds_df or 'id' not in preds_df:
        raise ValueError('Prediction file must contain id and label columns')

    merged = test_df.merge(preds_df, on='id', suffixes=('_true', '_pred'))
    y_true = merged['label_true']
    y_pred = merged['label_pred']

    cm = confusion_matrix(y_true, y_pred)
    labels = [id2label[i] for i in sorted(id2label.keys())]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap='Blues')  # blue colormap
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha='right')
    plt.yticks(ticks, labels)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')

    # Annotate counts
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                     color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate training and evaluation visualizations')
    parser.add_argument('--checkpoint_dir', '-c', required=True, help='Path to checkpoints')
    parser.add_argument('--test_file', '-t', required=True, help='Test JSONL with true labels')
    parser.add_argument('--pred_file', '-p', required=True, help='Predictions JSONL file')
    parser.add_argument('--output_dir', '-o', default='./plots', help='Directory to save plots')
    parser.add_argument('--id2label', '-m', help='JSON file mapping id to label', type=str)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # Load id2label mapping
    if args.id2label and os.path.exists(args.id2label):
        id2label = json.load(open(args.id2label))
    else:
        id2label = {0: 'human', 1: 'chatGPT', 2: 'cohere', 3: 'davinci', 4: 'bloomz', 5: 'dolly'}

    history_df = load_trainer_state(args.checkpoint_dir)

    # Generate plots
    plot_loss_by_epoch(history_df, args.output_dir)
    plot_eval_metrics(history_df, args.output_dir)
    plot_learning_rate(history_df, args.output_dir)
    plot_confusion(args.test_file, args.pred_file, id2label, args.output_dir)
    plot_classification_report(args.test_file, args.pred_file, id2label, args.output_dir)

    print(f"Plots saved to {args.output_dir}")


if __name__ == '__main__':
    main()
