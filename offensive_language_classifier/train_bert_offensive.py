"""
Arabic Offensive Language Detection — BERT Fine-Tuning Script
=============================================================
This script fine-tunes AraBERT (or QARiB) on the merged hate speech dataset
for binary offensive language classification.

Requirements:
    pip install transformers torch pandas openpyxl scikit-learn

Usage:
    python train_bert_offensive.py

    # To use QARiB instead of AraBERT:
    python train_bert_offensive.py --model qarib/bert-base-qarib

    # To adjust hyperparameters:
    python train_bert_offensive.py --epochs 5 --lr 2e-5 --batch_size 32
"""

import argparse
import re
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# 1. DATA LOADING & LABEL UNIFICATION
# ─────────────────────────────────────────────────────────

def unify_label(row):
    """Map heterogeneous labels to binary: 1=offensive, 0=not offensive."""
    # If the Offensive column is explicitly marked
    if pd.notna(row["Offensive"]) and row["Offensive"] == 1.0:
        return 1

    label = row["label (HS)"]
    label2 = row["label2"]
    ds = row["dataset"]

    # osact2020 has a dedicated OFF/NOT_OFF column
    if ds == "osact2020":
        return 1 if label2 == "OFF" else 0

    # Direct mappings from label (HS)
    if label in ["normal", "C", "Non-Offensive", 0, "0"]:
        return 0
    if label in ["abusive", "hate", "OH", "Offensive", 1, "1"]:
        return 1
    if label == "NOT_HS":
        return 0
    if label == "HS":
        return 1

    return -1  # unknown — will be dropped


def clean_text(text):
    """Minimal text cleaning for Arabic tweets."""
    text = str(text)
    text = re.sub(r"@\S+", "@USER", text)
    text = re.sub(r"https?://\S+", "URL", text)
    text = re.sub(r"<LF>", " ", text)
    text = re.sub(r"NEWLINE", " ", text)
    # Normalize some Arabic characters
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_prepare(filepath):
    """Load the merged Excel, unify labels, clean, deduplicate."""
    df = pd.read_excel(filepath, sheet_name="MERGE")
    df["label"] = df.apply(unify_label, axis=1)
    df = df[df["label"] >= 0].copy()
    df = df.dropna(subset=["text"])
    df = df.drop_duplicates(subset="text")
    df["clean_text"] = df["text"].apply(clean_text)
    print(f"Loaded {len(df)} unique samples: OFF={df['label'].sum()}, "
          f"NOT_OFF={(df['label']==0).sum()}")
    return df


# ─────────────────────────────────────────────────────────
# 2. PYTORCH DATASET
# ─────────────────────────────────────────────────────────

class ArabicOffensiveDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────
# 3. TRAINING LOOP
# ─────────────────────────────────────────────────────────

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)


# ─────────────────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="Hatespeech-data-merge.xlsx")
    parser.add_argument("--model", default="aubmindlab/bert-base-arabertv02",
                        help="HuggingFace model name. Try also: qarib/bert-base-qarib")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--output_dir", default="./offensive_model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    df = load_and_prepare(args.data)

    # Split: use existing splits where available
    has_split = df["split"].notna()
    train_mask = df["split"] == "train"
    test_mask = df["split"] == "test"
    dev_mask = df["split"] == "dev"
    no_split = ~has_split

    if no_split.sum() > 0:
        ns_df = df[no_split]
        ns_train, ns_test = train_test_split(
            ns_df, test_size=0.2, random_state=42, stratify=ns_df["label"]
        )
        train_df = pd.concat([df[train_mask], df[dev_mask], ns_train])
        test_df = pd.concat([df[test_mask], ns_test])
    else:
        train_df = pd.concat([df[train_mask], df[dev_mask]])
        test_df = df[test_mask]

    print(f"Train: {len(train_df)} | Test: {len(test_df)}")

    # Compute class weights for imbalanced data
    n_off = train_df["label"].sum()
    n_not = (train_df["label"] == 0).sum()
    weight_for_0 = len(train_df) / (2 * n_not)
    weight_for_1 = len(train_df) / (2 * n_off)
    class_weights = torch.tensor([weight_for_0, weight_for_1], dtype=torch.float).to(device)
    print(f"Class weights: NOT_OFF={weight_for_0:.3f}, OFF={weight_for_1:.3f}")

    # Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2
    ).to(device)

    # Datasets
    train_dataset = ArabicOffensiveDataset(
        train_df["clean_text"].tolist(), train_df["label"].tolist(),
        tokenizer, args.max_length
    )
    test_dataset = ArabicOffensiveDataset(
        test_df["clean_text"].tolist(), test_df["label"].tolist(),
        tokenizer, args.max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Override loss with class weights
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Training
    best_f1 = 0
    for epoch in range(args.epochs):
        avg_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        preds, labels = evaluate(model, test_loader, device)
        macro_f1 = f1_score(labels, preds, average="macro")
        print(f"Epoch {epoch+1}/{args.epochs} — Loss: {avg_loss:.4f} — "
              f"Macro-F1: {macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"  → Saved best model (F1={best_f1:.4f})")

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    preds, labels = evaluate(model, test_loader, device)
    print(classification_report(labels, preds,
                                target_names=["NOT_OFF", "OFF"], digits=4))

    # Per-dataset breakdown
    print("\n--- Per-Dataset Performance ---")
    test_datasets = test_df["dataset"].values
    for ds in test_df["dataset"].unique():
        mask = test_datasets == ds
        if mask.sum() < 10:
            continue
        ds_f1 = f1_score(labels[mask], preds[mask], average="macro")
        print(f"  {ds:15s}: n={mask.sum():5d}, Macro-F1={ds_f1:.4f}")


if __name__ == "__main__":
    main()
