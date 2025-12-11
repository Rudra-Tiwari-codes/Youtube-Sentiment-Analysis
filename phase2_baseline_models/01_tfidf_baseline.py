"""
Baseline Model: Simple Logistic Regression

This is our starting point - the simplest reasonable model.
Converts text to numbers using TF-IDF, then trains a logistic regression.
If our transformer model can't beat this, something's wrong!
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
import joblib
import time
from pathlib import Path
import sys
import yaml
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from config.yaml"""
    config_path = project_root / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data(config):
    """Load train, validation, and test sets"""
    # Loading the data like it's the Matrix - red pill or blue pill?
    logger.info("Loading datasets...")
    
    data_dir = Path(config['paths']['processed_data'])
    
    train = pd.read_csv(data_dir / 'train.csv')
    val = pd.read_csv(data_dir / 'val.csv')
    test = pd.read_csv(data_dir / 'test.csv')
    
    logger.info(f"Train: {len(train):,} records")
    logger.info(f"Val: {len(val):,} records")
    logger.info(f"Test: {len(test):,} records")
    
    return train, val, test


def prepare_features(train, val, test, config):
    """
    Convert text to numbers using TF-IDF
    TF-IDF looks at how important each word is in each comment
    Rare words get higher scores than common words like "the" or "is"
    """
    logger.info("Building TF-IDF features...")
    
    tfidf_config = config['models']['tfidf']
    
    vectorizer = TfidfVectorizer(
        max_features=tfidf_config['max_features'],
        ngram_range=tuple(tfidf_config['ngram_range']),
        min_df=tfidf_config['min_df'],
        max_df=tfidf_config['max_df'],
        stop_words='english',
        sublinear_tf=True  # Log scaling
    )
    
    # Fit on training data only
    X_train = vectorizer.fit_transform(train['cleaned_text'])
    X_val = vectorizer.transform(val['cleaned_text'])
    X_test = vectorizer.transform(test['cleaned_text'])
    
    logger.info(f"TF-IDF shape: {X_train.shape}")
    logger.info(f"Vocabulary size: {len(vectorizer.vocabulary_):,}")

    model_dir = Path(config['paths']['models'])
    model_dir.mkdir(exist_ok=True)
    joblib.dump(vectorizer, model_dir / 'tfidf_vectorizer.pkl')
    logger.info(" Saved TF-IDF vectorizer")
    
    return X_train, X_val, X_test, vectorizer


def prepare_labels(train, val, test):
    """Encode sentiment labels as integers"""
    logger.info("Encoding sentiment labels ")
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train['sentiment'])
    y_val = label_encoder.transform(val['sentiment'])
    y_test = label_encoder.transform(test['sentiment'])
    logger.info(f"Classes: {label_encoder.classes_}")
    return y_train, y_val, y_test, label_encoder


def train_baseline(X_train, y_train, config):
    """
    Train Logistic Regression baseline
    Hyperparameters:
    - L2 regularization (C=1.0)
    - LBFGS solver (good for multiclass)
    - Max 1000 iterations
    """
    logger.info("Training Logistic Regression baseline ")
    
    logreg_config = config['models']['logistic_regression']
    
    model = LogisticRegression(
        C=logreg_config['C'],
        max_iter=logreg_config['max_iter'],
        random_state=config['random_state'],
        solver='lbfgs',
        multi_class='multinomial',
        n_jobs=-1
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    logger.info(f" Training complete in {train_time:.2f}s")
    
    return model, train_time


def evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, config):
    """Model evaluation"""
    logger.info("\n" + "="*60)
    logger.info("BASELINE MODEL EVALUATION")
    logger.info("="*60)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Probabilities for ROC-AUC
    y_test_proba = model.predict_proba(X_test)
    
    results = {}
    
    # Training set metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='macro')
    
    # Validation set metrics
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='macro')
    
    # Test set metrics (final evaluation)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1_macro = f1_score(y_test, y_test_pred, average='macro')
    test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
    
    # ROC-AUC (one-vs-rest for multiclass)
    try:
        test_auc = roc_auc_score(y_test, y_test_proba, multi_class='ovr', average='macro')
    except:
        test_auc = None
    
    logger.info("\n ACCURACY SCORES")
    logger.info(f"Train Accuracy: {train_acc:.4f}")
    logger.info(f"Val Accuracy:   {val_acc:.4f}")
    logger.info(f"Test Accuracy:  {test_acc:.4f}")
    
    logger.info("\n F1 SCORES")
    logger.info(f"Train F1 (macro): {train_f1:.4f}")
    logger.info(f"Val F1 (macro):   {val_f1:.4f}")
    logger.info(f"Test F1 (macro):  {test_f1_macro:.4f}")
    logger.info(f"Test F1 (weighted): {test_f1_weighted:.4f}")
    
    if test_auc:
        logger.info(f"\n ROC-AUC (OVR): {test_auc:.4f}")
    
    # classification report
    logger.info("\n" + "="*60)
    logger.info("Classification (Test Set)")
    logger.info("="*60)
    report = classification_report(
        y_test, 
        y_test_pred, 
        target_names=label_encoder.classes_,
        digits=4
    )
    logger.info("\n" + report)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    logger.info("\n" + "="*60)
    logger.info("CONFUSION MATRIX (Test Set)")
    logger.info("="*60)
    logger.info(f"Classes: {label_encoder.classes_}")
    logger.info("\n" + str(cm))
    
    # Save results
    results = {
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'test_accuracy': test_acc,
        'train_f1_macro': train_f1,
        'val_f1_macro': val_f1,
        'test_f1_macro': test_f1_macro,
        'test_f1_weighted': test_f1_weighted,
        'test_auc': test_auc,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    # Save to file
    eval_dir = project_root / 'phase2_baseline_models' / 'evaluation'
    eval_dir.mkdir(exist_ok=True, parents=True)
    
    with open(eval_dir / 'baseline_results.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("TF-IDF + LOGISTIC REGRESSION BASELINE\n")
        f.write("="*60 + "\n\n")
        f.write(f"Train Accuracy: {train_acc:.4f}\n")
        f.write(f"Val Accuracy:   {val_acc:.4f}\n")
        f.write(f"Test Accuracy:  {test_acc:.4f}\n\n")
        f.write(f"Test F1 (macro):    {test_f1_macro:.4f}\n")
        f.write(f"Test F1 (weighted): {test_f1_weighted:.4f}\n")
        if test_auc:
            f.write(f"Test ROC-AUC (OVR): {test_auc:.4f}\n\n")
        f.write("\nCLASSIFICATION REPORT\n")
        f.write("="*60 + "\n")
        f.write(report)
        f.write("\n\nCONFUSION MATRIX\n")
        f.write("="*60 + "\n")
        f.write(str(cm))
    
    logger.info(f"\n Results saved to {eval_dir / 'baseline_results.txt'}")
    
    return results


def main():
    """Main execution pipeline"""
    logger.info(" Starting TF-IDF + Logistic Regression Baseline")
    logger.info("="*60)
    
    # Load config
    config = load_config()
    
    # Load data
    train, val, test = load_data(config)
    
    # Prepare features (TF-IDF)
    X_train, X_val, X_test, vectorizer = prepare_features(train, val, test, config)
    
    # Prepare labels
    y_train, y_val, y_test, label_encoder = prepare_labels(train, val, test)
    
    # Train baseline model
    model, train_time = train_baseline(X_train, y_train, config)
    
    # Save model
    model_dir = Path(config['paths']['models'])
    joblib.dump(model, model_dir / 'logreg_baseline.pkl')
    joblib.dump(label_encoder, model_dir / 'label_encoder.pkl')
    logger.info(" Saved baseline model and label encoder")
    
    # Evaluate model
    results = evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, config)
    
    # Inference speed test
    logger.info("\n" + "="*60)
    logger.info("INFERENCE SPEED TEST")
    logger.info("="*60)
    
    n_samples = 1000
    test_sample = X_test[:n_samples]
    
    start_time = time.time()
    _ = model.predict(test_sample)
    inference_time = time.time() - start_time
    
    time_per_pred = (inference_time / n_samples) * 1000  # ms
    
    logger.info(f"Predictions: {n_samples}")
    logger.info(f"Total time: {inference_time:.3f}s")
    logger.info(f"Time per prediction: {time_per_pred:.2f}ms")
 # Yeah just getting base info that we need for comparability + look professional   
    logger.info("\n" + "="*60)
    logger.info(" BASELINE COMPLETE")
    logger.info("="*60)
    logger.info(f"Test Accuracy: {results['test_accuracy']:.4f}")
    logger.info(f"Test F1 (macro): {results['test_f1_macro']:.4f}")
    logger.info(f"Training time: {train_time:.2f}s")
    logger.info(f"Inference: {time_per_pred:.2f}ms/prediction")
    logger.info("\nThis is the benchmark to beat with transformers! ")


if __name__ == '__main__':
    main()
