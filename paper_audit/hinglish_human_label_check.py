"""
How good are VADER labels (and models trained on them) against HUMAN labels on Hinglish?

The YouTube dataset itself has no human labels, so this uses a public, human-annotated
Hinglish sentiment set: the IIITH Codemixed dataset (Joshi, Prabhu, Shrivastava & Varma,
COLING 2016): 3,879 romanized Hindi-English Facebook comments, 0/1/2 = Neg/Neu/Pos.

Experiments (all scored against the human labels):
  1. Phase 1 language detector on this romanized Hinglish
  2. VADER labels, exactly as phase1_data_engineering/04_sentiment_labeling.py makes them
  3. The committed Phase 2 models (trained on VADER-labelled YouTube comments)
  4. Same classifier, trained on VADER labels vs trained on human labels (5-fold CV):
     what the paper's methodology would report vs what a human would measure

Usage:
    pip install scikit-learn==1.7.2 joblib numpy pandas vaderSentiment emoji langdetect
    python paper_audit/hinglish_human_label_check.py
"""

import importlib.util
import json
import urllib.request
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")
project_root = Path(__file__).parent.parent
out_dir = project_root / "paper_audit" / "results"
DATA_URL = "https://raw.githubusercontent.com/DrImpossible/Sub-word-LSTM/master/Data/IIITH_Codemixed.txt"
CLASSES = ["Negative", "Neutral", "Positive"]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dataset(cache):
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, cache)
    rows = [line.rstrip("\n").split("\t") for line in cache.read_text(encoding="utf-8").splitlines()]
    df = pd.DataFrame([r[:4] for r in rows if len(r) >= 4], columns=["id", "text", "label_a", "label_b"])
    df["human"] = df["label_a"].astype(int).map(dict(enumerate(CLASSES)))
    return df


def score(y_true, y_pred):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", labels=CLASSES) * 100, 2),
        "cohen_kappa": round(cohen_kappa_score(y_true, y_pred), 3),
        "per_class_f1": dict(zip(CLASSES, (f1_score(y_true, y_pred, average=None, labels=CLASSES) * 100)
                                 .round(2).tolist())),
        "confusion_matrix(rows=human Neg/Neu/Pos)": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist(),
    }


def main():
    out_dir.mkdir(parents=True, exist_ok=True)
    cleaner_mod = load_module(project_root / "phase1_data_engineering" / "02_clean_data.py", "clean")
    langdet_mod = load_module(project_root / "phase1_data_engineering" / "03_language_detection.py", "lang")

    df = load_dataset(project_root / "paper_audit" / "external_data" / "IIITH_Codemixed.txt")
    print(f"Loaded {len(df):,} human-labelled comments; "
          f"label columns agree on {(df.label_a == df.label_b).mean() * 100:.1f}%")

    # Same cleaning and language detection as the paper's Phase 1
    cleaner, detector = cleaner_mod.TextCleaner(), langdet_mod.LanguageDetector()
    df["cleaned_text"] = df["text"].map(cleaner.clean_text)
    df = df[df["cleaned_text"].str.strip() != ""].reset_index(drop=True)
    df["language"] = df["cleaned_text"].map(detector.detect_language)

    results = {"n": len(df), "human_label_distribution_pct":
               (df["human"].value_counts(normalize=True) * 100).round(2).to_dict()}
    results["phase1_language_detector_pct"] = (df["language"].value_counts(normalize=True) * 100).round(2).to_dict()

    # VADER exactly as 04_sentiment_labeling.py
    vader = SentimentIntensityAnalyzer()
    df["vader_compound"] = df["cleaned_text"].map(lambda t: vader.polarity_scores(t)["compound"])
    df["vader"] = np.select([df.vader_compound >= 0.05, df.vader_compound <= -0.05],
                            ["Positive", "Negative"], "Neutral")
    results["vader_label_distribution_pct"] = (df["vader"].value_counts(normalize=True) * 100).round(2).to_dict()
    results["vader_vs_human"] = score(df["human"], df["vader"])
    results["vader_compound_exactly_zero_pct"] = round((df.vader_compound == 0).mean() * 100, 2)
    results["majority_class_baseline_acc"] = round(df["human"].value_counts(normalize=True).iloc[0] * 100, 2)

    # Committed Phase 2 models, trained on VADER-labelled YouTube comments
    vectorizer = joblib.load(project_root / "models" / "tfidf_vectorizer.pkl")
    encoder = joblib.load(project_root / "models" / "label_encoder.pkl")
    X = vectorizer.transform(df["cleaned_text"])
    results["committed_models_vs_human"] = {}
    results["committed_models_vs_vader"] = {}
    for name in ["linear_svm", "logreg_baseline", "naive_bayes", "random_forest", "gradient_boosting"]:
        model = joblib.load(project_root / "models" / f"{name}.pkl")
        pred = encoder.inverse_transform(model.predict(X))
        results["committed_models_vs_human"][name] = score(df["human"], pred)
        results["committed_models_vs_vader"][name] = round(accuracy_score(df["vader"], pred) * 100, 2)
    oov = np.asarray(X.sum(axis=1)).ravel() == 0
    results["comments_with_zero_tfidf_features_pct"] = round(oov.mean() * 100, 2)

    # Same pipeline, two label sources, scored against humans (5-fold CV)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = {"trained_on_vader": np.empty(len(df), dtype=object),
             "trained_on_human": np.empty(len(df), dtype=object)}
    for train_idx, test_idx in cv.split(df, df["human"]):
        for key, label_col in [("trained_on_vader", "vader"), ("trained_on_human", "human")]:
            vec = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_df=0.95, sublinear_tf=True,
                                  max_features=10000)
            clf = LinearSVC(random_state=42)
            clf.fit(vec.fit_transform(df.loc[train_idx, "cleaned_text"]), df.loc[train_idx, label_col])
            preds[key][test_idx] = clf.predict(vec.transform(df.loc[test_idx, "cleaned_text"]))
    results["controlled_experiment_tfidf_svm_5fold"] = {
        "trained_on_vader__scored_vs_vader (what the paper reports)":
            round(accuracy_score(df["vader"], preds["trained_on_vader"]) * 100, 2),
        "trained_on_vader__scored_vs_human (real accuracy)": score(df["human"], preds["trained_on_vader"]),
        "trained_on_human__scored_vs_human": score(df["human"], preds["trained_on_human"]),
    }

    examples = df[(df.vader != df.human)].sample(15, random_state=0)[["cleaned_text", "human", "vader", "vader_compound"]]
    examples.to_csv(out_dir / "hinglish_vader_disagreements_sample.csv", index=False)
    (out_dir / "hinglish_human_label_check.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
