import argparse
import logging
import pandas as pd
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(
        description="Balance a JSONL dataset by label via undersampling/oversampling."
    )
    p.add_argument(
        "--input", "-i", required=True,
        help="Path to input JSONL file"
    )
    p.add_argument(
        "--output", "-o", required=True,
        help="Path where to write balanced JSONL"
    )
    p.add_argument(
        "--method", "-m", choices=["undersample", "oversample"], default="undersample",
        help="Whether to undersample majority classes or oversample minority classes"
    )
    p.add_argument(
        "--target", "-t", type=int, default=None,
        help="If set, sample every class to this exact count (overrides method)"
    )
    p.add_argument(
        "--seed", "-s", type=int, default=42,
        help="Random seed for reproducibility"
    )
    return p.parse_args()

def balance_df(df: pd.DataFrame, method: str, target: int, seed: int) -> pd.DataFrame:
    np.random.seed(seed)
    counts = df['label'].value_counts().to_dict()
    # Determine sampling size per label
    if target is not None:
        size_per_label = {lbl: target for lbl in counts}
    else:
        if method == "undersample":
            min_count = min(counts.values())
            size_per_label = {lbl: min_count for lbl in counts}
        else:  # oversample
            max_count = max(counts.values())
            size_per_label = {lbl: max_count for lbl in counts}

    logging.info(f"Original class counts: {counts}")
    logging.info(f"Sampling to:      {size_per_label}")

    # Sample each class
    balanced_parts = []
    for lbl, group in df.groupby('label'):
        n_desired = size_per_label[lbl]
        if len(group) == n_desired:
            sampled = group
        elif len(group) > n_desired:
            sampled = group.sample(n_desired, replace=False, random_state=seed)
        else:  # len(group) < n_desired
            sampled = group.sample(n_desired, replace=True, random_state=seed)
        balanced_parts.append(sampled)

    # Concatenate and shuffle
    balanced = pd.concat(balanced_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return balanced

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logging.info(f"Loading data from {args.input}")
    df = pd.read_json(args.input, lines=True)

    balanced = balance_df(df, args.method, args.target, args.seed)

    out_counts = balanced['label'].value_counts().to_dict()
    logging.info(f"Balanced class counts: {out_counts}")

    logging.info(f"Writing balanced dataset to {args.output}")
    balanced.to_json(args.output, orient="records", lines=True)
    logging.info("Done.")

if __name__ == "__main__":
    main()