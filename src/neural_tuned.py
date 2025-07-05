import os
import argparse
import logging
import re
import emoji

import pandas as pd
import numpy as np
import torch

from datasets import Dataset
from sklearn.model_selection import train_test_split
from scipy.special import softmax
import evaluate

from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.trainer_utils import set_seed
from transformers.trainer_callback import EarlyStoppingCallback



def clean_text(text: str) -> str:
    """
    Basic text cleaning for consistency across tokenizers.
    1) Lowercase
    2) Replace URLs with <URL>
    3) Remove HTML tags
    4) Strip @mentions and #hashtags
    5) Convert emojis to text
    6) Remove unwanted characters (keep alphanumeric and basic punctuation)
    7) Collapse multiple spaces
    8) Handle empty strings
    """
    # text = text.lower()
    text = re.sub(r'https?://\S+', ' <URL> ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'@\w+|#\w+', ' ', text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r'[^a-z0-9<>\[\]_.,!?\s]', ' ', text)
    text = " ".join(text.split()).strip()
    return text if text else "[BLANK]"


def preprocess_function(examples, tokenizer, max_length: int = 256):
    """
    Tokenize texts, truncating/padding to max_length.
    """
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


def get_data(train_path: str, test_path: str, random_seed: int):
    """
    1) Load JSONL train/test into Pandas DataFrames.
    2) Clean the `text` column.
    3) Split train into train+validation stratified by label.
    """
    train_df = pd.read_json(train_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)

    # Clean text
    train_df["text"] = train_df["text"].map(clean_text)
    test_df["text"] = test_df["text"].map(clean_text)

    # Stratified split: 80% train / 20% dev
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.20,
        stratify=train_df["label"],
        random_state=random_seed,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def compute_metrics(eval_pred):
    """
    Compute micro-F1 on multi-class (6 labels).
    """
    f1_metric = evaluate.load("f1")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    result = f1_metric.compute(predictions=preds, references=labels, average="micro")
    if result is None:
        raise ValueError
    return result


def fine_tune(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    output_dir: str,
    id2label: dict,
    label2id: dict,
    model_name: str,
    device: torch.device,
    num_epochs: int = 5,
    per_device_batch: int = 8,
    gradient_accumulation_steps: int = 2,
    max_length: int = 256,
):
    """
    Fine-tune a transformer for 6-way classification with:
    - fp16
    - gradient checkpointing
    - label smoothing
    - warmup ratio
    - early stopping
    """
    # Convert to HF Datasets
    train_ds = Dataset.from_pandas(train_df)
    valid_ds = Dataset.from_pandas(valid_df)

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)

    # Tokenize dataset
    tokenized_train = train_ds.map(
        lambda ex: preprocess_function(ex, tokenizer, max_length),
        batched=True,
        remove_columns=["text"],
    )
    tokenized_valid = valid_ds.map(
        lambda ex: preprocess_function(ex, tokenizer, max_length),
        batched=True,
        remove_columns=["text"],
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    # Prepare TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=per_device_batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        warmup_ratio=0.10,
        fp16=True,
        gradient_checkpointing=True,
        label_smoothing_factor=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_total_limit=2,
        eval_accumulation_steps=1,
        dataloader_num_workers=4,
        logging_steps=50,
        logging_dir=os.path.join(output_dir, "logs"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Start training
    trainer.train()

    # Save the best model to `<output_dir>/best/`
    best_dir = os.path.join(output_dir, "best")
    os.makedirs(best_dir, exist_ok=True)
    trainer.save_model(best_dir)


def test(
    test_df: pd.DataFrame,
    model_path: str,
    id2label: dict,
    label2id: dict,
    device: torch.device,
    max_length: int = 256,
):
    """
    Run inference on the test set, return classification report and predictions.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)

    test_ds = Dataset.from_pandas(test_df)
    tokenized_test = test_ds.map(
        lambda ex: preprocess_function(ex, tokenizer, max_length),
        batched=True,
        remove_columns=["text"],
    )

    data_collator = DataCollatorWithPadding(tokenizer)
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    predictions_output = trainer.predict(tokenized_test) # type: ignore[arg-type]
    logits = predictions_output.predictions
    probs = softmax(logits, axis=-1)
    preds = np.argmax(probs, axis=1)

    # If labels exist, compute detailed report
    if predictions_output.label_ids is not None:
        report = evaluate.load("bstrai/classification_report")
        results = report.compute(
            predictions=preds, references=predictions_output.label_ids
        )
    else:
        results = None

    return results, preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a compact transformer for Subtask B")
    parser.add_argument("--train_file_path", "-tr", required=True, help="Path to the train JSONL file", type=str)
    parser.add_argument("--test_file_path", "-t", required=True, help="Path to the test JSONL file", type=str)
    parser.add_argument("--model", "-m", required=False, default="xlm-roberta-base", help="HuggingFace model name (e.g., distilroberta-base, microsoft/MiniLM-L6-H384-uncased, albert-base-v2)", type=str)
    parser.add_argument("--output_dir", "-o", required=False, default="./checkpoints", help="Directory to save checkpoints and best model", type=str)
    parser.add_argument("--prediction_file_path", "-p", required=False, default="./subtaskB_predictions.jsonl", help="Where to save test predictions (JSONL)", type=str)
    parser.add_argument("--num_epochs", "-e", required=False, default=5, help="Number of training epochs", type=int)
    parser.add_argument("--batch_size", "-b", required=False, default=8, help="Per-device batch size", type=int)
    parser.add_argument("--accum_steps", "-a", required=False, default=2, help="Gradient accumulation steps (to simulate larger batches)", type=int)
    parser.add_argument("--max_length", "-l", required=False, default=256, help="Max sequence length for tokenization", type=int)
    parser.add_argument("--seed", "-s", required=False, default=0, help="Random seed", type=int)

    args = parser.parse_args()
    set_seed(args.seed)

    # Map labels for Subtask B
    id2label = {0: "human", 1: "chatGPT", 2: "cohere", 3: "davinci", 4: "bloomz", 5: "dolly"}
    label2id = {v: k for k, v in id2label.items()}

    # Device check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Validate file paths
    if not os.path.exists(args.train_file_path):
        raise FileNotFoundError(f"Train file not found: {args.train_file_path}")
    if not os.path.exists(args.test_file_path):
        raise FileNotFoundError(f"Test file not found: {args.test_file_path}")

    # Prepare data
    train_df, valid_df, test_df = get_data(
        args.train_file_path, args.test_file_path, random_seed=args.seed
    )

    # Fine-tune
    ckpt_dir = os.path.join(args.output_dir, f"{args.model.replace('/', '_')}_subtaskB")
    os.makedirs(ckpt_dir, exist_ok=True)
    fine_tune(
        train_df=train_df,
        valid_df=valid_df,
        output_dir=ckpt_dir,
        id2label=id2label,
        label2id=label2id,
        model_name=args.model,
        device=device,
        num_epochs=args.num_epochs,
        per_device_batch=args.batch_size,
        gradient_accumulation_steps=args.accum_steps,
        max_length=args.max_length,
    )

    # Test
    best_model_dir = os.path.join(ckpt_dir, "best")
    results, preds = test(
        test_df=test_df,
        model_path=best_model_dir,
        id2label=id2label,
        label2id=label2id,
        device=device,
        max_length=args.max_length,
    )

    # Log metrics
    if results is not None:
        logging.info("=== Classification Report ===")
        for k, v in results.items():
            logging.info(f"{k}: {v}")

    # Save predictions
    pd.DataFrame({"id": test_df["id"], "label": preds}).to_json(
        args.prediction_file_path, orient="records", lines=True
    )
    print(f"Done. Predictions saved to {args.prediction_file_path}")
