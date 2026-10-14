"""
Reproduce the paper-audit evidence using ONLY files committed to this repo.

No dataset or GPU needed. Checks:
  1. VADER (the label source) on Hindi / Hinglish text and on Hindi words
     that collide with English lexicon entries.
  2. Label distribution per language, read from the Phase 1 EDA figure data.
  3. Comment date range, read from the Phase 1 temporal figure data.
  4. Top features of the committed Linear SVM (exposes VADER artifacts).
  5. TF-IDF config (English stop words remove negations).
  6. Metrics recomputed from the confusion matrices in the results files.

Usage:
    pip install scikit-learn==1.7.2 joblib numpy vaderSentiment
    python paper_audit/verify_from_artifacts.py
"""

import base64
import collections
import json
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")
project_root = Path(__file__).parent.parent
eda_dir = project_root / "phase1_data_engineering" / "figures"


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def vader_label(compound):
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


def load_plotly_traces(html_path):
    """Pull the trace list out of a plotly-generated HTML file."""
    html = html_path.read_text(encoding="utf-8")
    match = re.search(r'Plotly\.newPlot\(\s*"[^"]+",\s*(\[.*?\])\s*,\s*\{', html, re.S)
    return json.loads(match.group(1))


def decode(values):
    """Plotly >= 6 stores numeric arrays as base64 'bdata'."""
    if isinstance(values, dict) and "bdata" in values:
        arr = np.frombuffer(base64.b64decode(values["bdata"]), dtype=values["dtype"])
        if isinstance(values.get("shape"), str):
            arr = arr.reshape([int(i) for i in values["shape"].split(",")])
        return arr.tolist()
    return values


def check_vader():
    section("1. VADER on Hindi / Hinglish (VADER produced every training label)")
    analyzer = SentimentIntensityAnalyzer()

    print("\nHindi words that collide with English VADER lexicon entries:")
    collisions = {
        "kia": "Hindi 'kiya' = did (VADER: 'killed in action')",
        "bad": "Hindi 'baad' = after",
        "ache": "Hindi 'acche' = good",
        "lag": "Hindi 'lag' = seems",
        "ha": "Hindi 'haan' = yes",
    }
    for word, meaning in collisions.items():
        print(f"  {word:6} valence={analyzer.lexicon.get(word):+.1f}   {meaning}")

    print("\nHindi sentiment words missing from the VADER lexicon:")
    missing = ["bekar", "badhiya", "ghatiya", "bakwas", "accha", "berozgari", "dhoka"]
    print("  " + ", ".join(w for w in missing if w not in analyzer.lexicon))

    samples = [
        ("ye sarkar bilkul bekar hai, koi naukri nahi", "Negative"),
        ("bahut badhiya video bhai, dil se dhanyavad", "Positive"),
        ("berozgari bahut badh gayi hai, ghar chalana mushkil hai", "Negative"),
        ("ghatiya policy, sab bakwas", "Negative"),
        ("accha laga sunke ki jobs aa rahi hai", "Positive"),
        ("बहुत अच्छा वीडियो है", "Positive"),
        ("सरकार ने युवाओं को धोखा दिया", "Negative"),
        ("Exam ke bad result kab aayega", "Neutral"),
    ]
    print("\nVADER labels for hand-written examples (expected = human reading):")
    wrong = 0
    for text, expected in samples:
        compound = analyzer.polarity_scores(text)["compound"]
        label = vader_label(compound)
        wrong += label != expected
        flag = "" if label == expected else "  <-- WRONG"
        print(f"  {compound:+.3f} {label:8} expected {expected:8} | {text}{flag}")
    print(f"  VADER wrong on {wrong}/{len(samples)} examples")


def check_language_label_distribution():
    section("2. Label distribution by language (from language_sentiment_heatmap.html)")
    trace = load_plotly_traces(eda_dir / "language_sentiment_heatmap.html")[0]
    classes, languages = trace["x"], trace["y"]
    pct = decode(trace["z"])
    counts = [[int(n) for n in row] for row in decode(trace["text"])]
    print(f"  {'Language':<12}" + "".join(f"{c:>18}" for c in classes) + f"{'Total':>9}")
    for lang, row_pct, row_n in zip(languages, pct, counts):
        cells = "".join(f"{n:>9,} ({p:5.1f}%)" for n, p in zip(row_n, row_pct))
        print(f"  {lang:<12}{cells}{sum(row_n):>9,}")

    trace = load_plotly_traces(eda_dir / "language_distribution.html")[0]
    values = [int(n) for n in decode(trace["values"])]
    total = sum(values)
    print("\n  Language detector output (Latin-script Hinglish is counted as 'English'):")
    for lang, n in zip(trace["labels"], values):
        print(f"    {lang:<12}{n:>9,} ({n / total * 100:5.2f}%)")


def check_date_range():
    section("3. Comment dates (from temporal_trends.html)")
    trace = load_plotly_traces(eda_dir / "temporal_trends.html")[0]
    per_year = collections.Counter()
    for month, n in zip(trace["x"], decode(trace["y"])):
        per_year[month[:4]] += int(n)
    total = sum(per_year.values())
    print(f"  First month: {trace['x'][0]}   Last month: {trace['x'][-1]}   Total: {total:,}")
    for year, n in sorted(per_year.items()):
        print(f"    {year}: {n:>7,} ({n / total * 100:5.2f}%)")
    before_2019 = sum(n for y, n in per_year.items() if y < "2019")
    print(f"  Before 2019: {before_2019:,} ({before_2019 / total * 100:.2f}%)")


def check_svm_features():
    section("4. Committed Linear SVM: top features per class")
    vectorizer = joblib.load(project_root / "models" / "tfidf_vectorizer.pkl")
    svm = joblib.load(project_root / "models" / "linear_svm.pkl")
    vocab = np.array(vectorizer.get_feature_names_out())
    for i, cls in enumerate(["Negative", "Neutral", "Positive"]):
        top = vocab[np.argsort(svm.coef_[i])[::-1][:20]]
        print(f"  {cls}: {', '.join(top)}")
    print("\n  Note 'kia', 'lag', 'ke bad', 'ache' under Negative: VADER lexicon")
    print("  collisions with Hindi words, which the classifier learned faithfully.")

    section("5. TF-IDF configuration actually used")
    print(f"  {vectorizer}")
    negators = ["not", "no", "never", "nothing", "nor", "none", "cannot", "without"]
    removed = [w for w in negators if w in ENGLISH_STOP_WORDS]
    print(f"  Negations deleted by stop_words='english': {removed}")
    devanagari = [t for t in vocab if re.search("[ऀ-ॿ]", t)]
    print(f"  Devanagari n-grams in vocabulary: {len(devanagari)} / {len(vocab)}")
    print(f"  Logistic Regression: {joblib.load(project_root / 'models' / 'logreg_baseline.pkl')}")
    print(f"  Linear SVM:          {svm}  (default loss = squared_hinge)")


def metrics_from_cm(cm):
    cm = np.array(cm, dtype=float)
    precision = np.diag(cm) / cm.sum(axis=0)
    recall = np.diag(cm) / cm.sum(axis=1)
    f1 = 2 * precision * recall / (precision + recall)
    support = cm.sum(axis=1)
    acc = np.trace(cm) / cm.sum()
    return acc, f1.mean(), (f1 * support).sum() / support.sum(), precision, recall, f1


def check_confusion_matrices():
    section("6. Metrics recomputed from confusion matrices in results files")
    matrices = {
        "DistilBERT Colab run (checkpoints/distilbert/distilbert_results.txt)":
            [[1932, 37, 119], [63, 5552, 106], [96, 44, 5212]],
        "DistilBERT older run (evaluation/distilbert_results.txt)":
            [[1580, 264, 244], [70, 5519, 132], [172, 380, 4800]],
        "'MuRIL' (evaluation/muril_results.txt)":
            [[1580, 264, 244], [70, 5519, 132], [172, 379, 4801]],
        "Logistic Regression (phase2 baseline_results.txt)":
            [[1345, 388, 355], [53, 5519, 149], [182, 551, 4619]],
    }
    for name, cm in matrices.items():
        acc, f1m, f1w, p, r, f1 = metrics_from_cm(cm)
        n = np.sum(cm)
        ci = 1.96 * np.sqrt(acc * (1 - acc) / n)
        print(f"\n  {name}")
        print(f"    acc={acc * 100:.2f}% (95% CI +/- {ci * 100:.2f} pp)  "
              f"F1-macro={f1m * 100:.2f}%  F1-weighted={f1w * 100:.2f}%")
        print("    per-class F1 (Neg/Neu/Pos): " + " / ".join(f"{x * 100:.2f}" for x in f1))
    diff = np.abs(np.array(matrices["DistilBERT older run (evaluation/distilbert_results.txt)"])
                  - np.array(matrices["'MuRIL' (evaluation/muril_results.txt)"])).sum()
    print(f"\n  Older DistilBERT vs 'MuRIL' differ in only {diff} cell-counts out of 13,161")
    print(f"  Majority-class (always 'Neutral') accuracy: {5721 / 13161 * 100:.2f}%")

    svm_acc, svm_f1m = 0.8857229693792265, 0.8588704661669165
    bert_acc, bert_f1m = 0.9647, 0.9563
    print(f"\n  DistilBERT vs Linear SVM: +{(bert_acc - svm_acc) * 100:.2f} pp accuracy "
          f"(relative +{(bert_acc / svm_acc - 1) * 100:.2f}%), "
          f"+{(bert_f1m - svm_f1m) * 100:.2f} pp macro-F1")
    print(f"  Negative-class F1 gain over LogReg: 73.34 -> 92.46 = +{92.46 - 73.34:.2f} pp "
          f"(relative +{(92.46 / 73.34 - 1) * 100:.1f}%)")


if __name__ == "__main__":
    check_vader()
    check_language_label_distribution()
    check_date_range()
    check_svm_features()
    check_confusion_matrices()
