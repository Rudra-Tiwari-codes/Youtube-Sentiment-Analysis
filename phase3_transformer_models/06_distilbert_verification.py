"""
DistilBERT re-run that produces the numbers the paper needs but the repo lacks.

Same training setup as Phase3_DistilBERT_Colab.ipynb (lr 2e-5, batch 32,
3 epochs, warmup 500, weight decay 0.01, max_length 256, best checkpoint by
validation macro-F1), plus:

  - several seeds, reported as mean +/- std
  - saved per-comment test predictions
  - bootstrap 95% confidence intervals
  - accuracy by language, including Latin-script (romanized) Hinglish
  - accuracy by VADER confidence (labels ARE VADER output, so this shows
    how much of the score is just re-learning VADER's threshold)
  - train/test exact-duplicate leakage check
  - majority-class and per-language majority baselines
  - a stratified sample for human annotation, and an evaluator for it

Run on a GPU (Colab T4: ~30 min per seed):
    python phase3_transformer_models/06_distilbert_verification.py --seeds 42 43 44

After hand-labelling the exported sample (fill the human_label column):
    python phase3_transformer_models/06_distilbert_verification.py \
        --evaluate-human results/verification/human_annotation_sample.csv

--smoke-test trains a tiny randomly initialised model on synthetic data to
check the pipeline end to end without a GPU or a HuggingFace download.
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

project_root = Path(__file__).parent.parent
CLASSES = ["Negative", "Neutral", "Positive"]
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}

# Common Hindi function words written in Latin script. A Latin-script comment
# with two or more of these is treated as romanized Hinglish; the Phase 1
# script detector labels all such comments 'English'.
HINGLISH_MARKERS = {
    "hai", "hain", "nahi", "nhi", "kya", "kyu", "kyun", "mein", "mai", "hum",
    "aur", "ka", "ki", "ke", "ko", "se", "ye", "yeh", "wo", "woh", "bhi",
    "tha", "thi", "kar", "karo", "raha", "rahe", "hota", "hoga", "sab",
    "bhai", "ji", "koi", "kuch", "bahut", "bohot", "acha", "accha", "sarkar",
    "naukri", "log", "logo", "apne", "unka", "iska", "jab", "tab", "par",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(project_root / "data" / "processed"))
    parser.add_argument("--out-dir", default=str(project_root / "results" / "verification"))
    parser.add_argument("--model", default="distilbert-base-uncased",
                        help="e.g. distilbert-base-uncased, google/muril-base-cased, xlm-roberta-base")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--annotation-sample-size", type=int, default=300)
    parser.add_argument("--evaluate-human", metavar="CSV",
                        help="Score VADER and the model against a hand-labelled sample, then exit")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_synthetic_split(n, seed):
    rng = np.random.default_rng(seed)
    words = {
        "Negative": ["worst", "bad", "fake", "scam", "fail"],
        "Neutral": ["exam", "date", "form", "result", "kab"],
        "Positive": ["good", "great", "thanks", "best", "nice"],
    }
    rows = []
    for _ in range(n):
        label = rng.choice(CLASSES, p=[0.16, 0.44, 0.40])
        hinglish = rng.random() < 0.3
        text = " ".join(rng.choice(words[label], 3)) + (" bhai ye sab kya hai" if hinglish else " job")
        rows.append({
            "cleaned_text": text,
            "sentiment": label,
            "language": rng.choice(["English", "Hindi", "Code-Mixed"], p=[0.9, 0.05, 0.05]),
            "vader_compound": {"Negative": -0.5, "Neutral": 0.0, "Positive": 0.5}[label]
                              + rng.normal(0, 0.2),
        })
    return pd.DataFrame(rows)


def load_splits(args):
    if args.smoke_test:
        train, val, test = (make_synthetic_split(n, s) for n, s in [(600, 0), (150, 1), (150, 2)])
        test = pd.concat([test, train.head(20)], ignore_index=True)  # exercise the leakage check
        return train, val, test
    data_dir = Path(args.data_dir)
    splits = []
    for name in ["train", "val", "test"]:
        path = data_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run phase1_data_engineering/05_train_test_split.py "
                                    "or upload train.csv / val.csv / test.csv into --data-dir.")
        df = pd.read_csv(path)
        df["cleaned_text"] = df["cleaned_text"].fillna("").astype(str)
        splits.append(df)
    return tuple(splits)


def is_romanized_hinglish(text):
    if re.search(r"[ऀ-ॿ]", text):
        return False
    tokens = re.findall(r"[a-z]+", text.lower())
    return sum(t in HINGLISH_MARKERS for t in tokens) >= 2


def add_analysis_columns(df):
    df = df.copy()
    df["romanized_hinglish"] = df["cleaned_text"].map(is_romanized_hinglish)
    df["language_refined"] = np.where(
        (df["language"] == "English") & df["romanized_hinglish"], "Romanized-Hinglish", df["language"])
    if "vader_compound" in df:
        bins = [-np.inf, 0.05, 0.2, 0.5, np.inf]
        names = ["|c|<0.05 (Neutral band)", "0.05-0.2 (near threshold)", "0.2-0.5", ">0.5 (confident)"]
        df["vader_confidence"] = pd.cut(df["vader_compound"].abs(), bins=bins, labels=names, right=False)
    return df


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model_and_tokenizer(args):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    id2label = dict(enumerate(CLASSES))
    if args.smoke_test:
        from transformers import DistilBertConfig, DistilBertForSequenceClassification

        local = project_root / "phase3_transformer_models" / "checkpoints" / "distilbert"
        tokenizer = AutoTokenizer.from_pretrained(str(local))
        config = DistilBertConfig(vocab_size=tokenizer.vocab_size, n_layers=2, n_heads=2, dim=64,
                                  hidden_dim=128, num_labels=3, id2label=id2label, label2id=LABEL2ID)
        return DistilBertForSequenceClassification(config), tokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=3, id2label=id2label, label2id=LABEL2ID)
    return model, tokenizer


def to_dataset(df, tokenizer, max_length):
    from datasets import Dataset

    ds = Dataset.from_dict({"text": df["cleaned_text"].tolist(),
                            "label": df["sentiment"].map(LABEL2ID).tolist()})
    ds = ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=max_length),
                batched=True, remove_columns=["text"])
    return ds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
            "f1_weighted": f1_score(labels, preds, average="weighted")}


def train_and_predict(args, seed, train_df, val_df, test_df, run_dir):
    import torch
    from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

    set_seed(seed)
    model, tokenizer = build_model_and_tokenizer(args)
    train_ds, val_ds, test_ds = (to_dataset(d, tokenizer, args.max_length)
                                 for d in (train_df, val_df, test_df))

    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_steps=0 if args.smoke_test else 500,
        logging_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=seed,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics)

    start = time.time()
    trainer.train()
    train_seconds = time.time() - start

    # Real per-epoch history, for the paper's training-curve figure.
    pd.DataFrame(trainer.state.log_history).to_csv(run_dir / "training_log_history.csv", index=False)

    output = trainer.predict(test_ds)
    probs = torch.softmax(torch.tensor(output.predictions), dim=1).numpy()
    return probs, train_seconds


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def bootstrap_ci(y_true, y_pred, metric, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = [metric(y_true[idx], y_pred[idx]) for idx in (rng.integers(0, n, n) for _ in range(n_boot))]
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def group_table(df, column):
    rows = []
    for group, g in df.groupby(column, observed=True):
        majority = g["sentiment"].value_counts(normalize=True)
        rows.append({
            column: group,
            "n": len(g),
            "accuracy": accuracy_score(g["sentiment"], g["predicted"]),
            "f1_macro": f1_score(g["sentiment"], g["predicted"], average="macro", labels=CLASSES,
                                 zero_division=0),
            "majority_label": majority.index[0],
            "majority_baseline_acc": majority.iloc[0],
            **{f"pct_{c}": (g["sentiment"] == c).mean() for c in CLASSES},
        })
    return pd.DataFrame(rows)


def analyse_run(test_df, probs, train_texts):
    df = test_df.copy()
    df["predicted"] = [CLASSES[i] for i in probs.argmax(axis=1)]
    for i, c in enumerate(CLASSES):
        df[f"prob_{c}"] = probs[:, i]
    df["text_in_train"] = df["cleaned_text"].isin(train_texts)

    y_true = df["sentiment"].map(LABEL2ID).to_numpy()
    y_pred = df["predicted"].map(LABEL2ID).to_numpy()
    acc_ci = bootstrap_ci(y_true, y_pred, accuracy_score)
    f1_ci = bootstrap_ci(y_true, y_pred, lambda a, b: f1_score(a, b, average="macro"))

    leak = df["text_in_train"]
    summary = {
        "n_test": len(df),
        "accuracy": accuracy_score(y_true, y_pred),
        "accuracy_ci95": acc_ci,
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_macro_ci95": f1_ci,
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "per_class": classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True,
                                           zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "majority_class_baseline_acc": float(df["sentiment"].value_counts(normalize=True).iloc[0]),
        "test_texts_also_in_train_pct": float(leak.mean() * 100),
        "accuracy_on_texts_in_train": float((df.loc[leak, "sentiment"] == df.loc[leak, "predicted"]).mean())
                                      if leak.any() else None,
        "accuracy_on_unseen_texts": float((df.loc[~leak, "sentiment"] == df.loc[~leak, "predicted"]).mean()),
    }
    tables = {
        "by_language": group_table(df, "language"),
        "by_language_refined": group_table(df, "language_refined"),
    }
    if "vader_confidence" in df:
        tables["by_vader_confidence"] = group_table(df, "vader_confidence")
    return df, summary, tables


def export_annotation_sample(test_df, n, out_dir):
    """Stratified by refined language so Hindi / Hinglish are not drowned out."""
    per_group = max(1, n // test_df["language_refined"].nunique())
    picked = [g.sample(min(len(g), per_group), random_state=0)
              for _, g in test_df.groupby("language_refined")]
    sample = pd.concat(picked).sample(frac=1, random_state=0)  # shuffle so annotators can't infer anything
    cols = ["cleaned_text", "language_refined", "sentiment"]
    out = sample[cols].rename(columns={"sentiment": "vader_label"})
    out.insert(1, "human_label", "")
    path = out_dir / "human_annotation_sample.csv"
    out.to_csv(path, index=True, index_label="row_id")
    return path


def evaluate_human(csv_path, out_dir):
    ann = pd.read_csv(csv_path)
    ann = ann[ann["human_label"].isin(CLASSES)]
    if ann.empty:
        raise SystemExit("No rows with human_label in {Negative, Neutral, Positive}.")
    report = {"n_annotated": len(ann)}
    pred_paths = sorted(out_dir.glob("test_predictions_seed*.csv"))
    if pred_paths:
        preds = pd.read_csv(pred_paths[0], index_col=0)["predicted"]
        ann["model_label"] = ann["row_id"].map(preds)
    for source in ["vader_label", "model_label"]:
        if source in ann:
            report[source] = {
                "accuracy_vs_human": accuracy_score(ann["human_label"], ann[source]),
                "cohen_kappa_vs_human": cohen_kappa_score(ann["human_label"], ann[source]),
                "f1_macro_vs_human": f1_score(ann["human_label"], ann[source], average="macro"),
                "accuracy_by_language": {
                    lang: accuracy_score(g["human_label"], g[source])
                    for lang, g in ann.groupby("language_refined")},
            }
    (out_dir / "human_evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def to_markdown(df):
    return df.to_markdown(index=False, floatfmt=".4f")


def write_report(out_dir, args, runs, tables_seed0, leakage_counts):
    accs = np.array([r["accuracy"] for r in runs])
    f1s = np.array([r["f1_macro"] for r in runs])
    first = runs[0]
    lines = [
        f"# Verification run: {args.model}",
        "",
        f"Seeds: {args.seeds}  |  epochs {args.epochs}  |  batch {args.batch_size}  |  lr {args.lr}  |  "
        f"max_length {args.max_length}" + ("  |  **SMOKE TEST (synthetic data, random weights)**"
                                          if args.smoke_test else ""),
        "",
        "## Headline (test set; labels are VADER output, not human judgement)",
        "",
        f"- Accuracy: {accs.mean() * 100:.2f}% +/- {accs.std() * 100:.2f} (mean +/- std over {len(runs)} seed(s))",
        f"- Macro-F1: {f1s.mean() * 100:.2f}% +/- {f1s.std() * 100:.2f}",
        f"- Seed {args.seeds[0]} accuracy 95% bootstrap CI: "
        f"{first['accuracy_ci95'][0] * 100:.2f}-{first['accuracy_ci95'][1] * 100:.2f}%",
        f"- Majority-class baseline: {first['majority_class_baseline_acc'] * 100:.2f}%",
        "",
        "## Train/test leakage",
        "",
        f"- Test comments whose exact text also appears in train: {first['test_texts_also_in_train_pct']:.2f}%"
        f" (train has {leakage_counts['train_duplicate_texts']:,} duplicated texts)",
        f"- Accuracy on those: {first['accuracy_on_texts_in_train']}",
        f"- Accuracy on unseen texts: {first['accuracy_on_unseen_texts']:.4f}",
        "",
    ]
    titles = {
        "by_language": "Accuracy by language (Phase 1 detector)",
        "by_language_refined": "Accuracy by language, romanized Hinglish split out of 'English'",
        "by_vader_confidence": "Accuracy by VADER |compound| (how hard each label was for VADER itself)",
    }
    for key, table in tables_seed0.items():
        lines += [f"## {titles[key]}", "", to_markdown(table), ""]
    lines += ["## Per-seed", "", to_markdown(pd.DataFrame([
        {"seed": str(s), "accuracy": r["accuracy"], "f1_macro": r["f1_macro"], "f1_weighted": r["f1_weighted"],
         "train_minutes": r["train_seconds"] / 60} for s, r in zip(args.seeds, runs)])), ""]
    (out_dir / "VERIFICATION_REPORT.md").write_text("\n".join(lines))


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.smoke_test:
        out_dir = out_dir.parent / (out_dir.name + "_smoke_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.evaluate_human:
        evaluate_human(args.evaluate_human, out_dir)
        return

    train_df, val_df, test_df = load_splits(args)
    test_df = add_analysis_columns(test_df)
    train_texts = set(train_df["cleaned_text"])
    leakage_counts = {"train_duplicate_texts": int(train_df["cleaned_text"].duplicated().sum())}
    print(f"Train {len(train_df):,} | Val {len(val_df):,} | Test {len(test_df):,}")

    runs, tables_seed0 = [], None
    for seed in args.seeds:
        print(f"\n=== Seed {seed} ===")
        run_dir = out_dir / f"seed{seed}"
        run_dir.mkdir(exist_ok=True)
        probs, train_seconds = train_and_predict(args, seed, train_df, val_df, test_df, run_dir)
        pred_df, summary, tables = analyse_run(test_df, probs, train_texts)
        summary["train_seconds"] = train_seconds
        pred_df.to_csv(out_dir / f"test_predictions_seed{seed}.csv")
        (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
        for name, table in tables.items():
            table.to_csv(run_dir / f"{name}.csv", index=False)
        print(f"accuracy {summary['accuracy']:.4f}  f1_macro {summary['f1_macro']:.4f}")
        runs.append(summary)
        tables_seed0 = tables_seed0 or tables

    export_annotation_sample(test_df, args.annotation_sample_size, out_dir)
    write_report(out_dir, args, runs, tables_seed0, leakage_counts)
    print(f"\nDone. Read {out_dir / 'VERIFICATION_REPORT.md'}")


if __name__ == "__main__":
    main()
