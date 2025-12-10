"""
Try out different machine learning models.

Now that we have a baseline, let's try other algorithms:
- Random Forest
- Gradient Boosting 
- Naive Bayes
- Support Vector Machine (SVM)

We'll compare them all and see which one works best on our TF-IDF features
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    roc_auc_score
)
import joblib
import time
from pathlib import Path
import yaml
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

project_root = Path(__file__).parent.parent


def load_config():
    """Load configuration"""
    config_path = project_root / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_features():
    """Load pre-computed TF-IDF features and labels"""
    logger.info("Loading TF-IDF features from baseline...")
    
    model_dir = project_root / 'models'
    data_dir = project_root / 'data' / 'processed'
    
    # Load vectorizer
    vectorizer = joblib.load(model_dir / 'tfidf_vectorizer.pkl')
    label_encoder = joblib.load(model_dir / 'label_encoder.pkl')
    
    # Load data
    train = pd.read_csv(data_dir / 'train.csv')
    val = pd.read_csv(data_dir / 'val.csv')
    test = pd.read_csv(data_dir / 'test.csv')
    
    # Transform text
    X_train = vectorizer.transform(train['cleaned_text'])
    X_val = vectorizer.transform(val['cleaned_text'])
    X_test = vectorizer.transform(test['cleaned_text'])
    
    # Encode labels
    y_train = label_encoder.transform(train['sentiment'])
    y_val = label_encoder.transform(val['sentiment'])
    y_test = label_encoder.transform(test['sentiment'])
    
    logger.info(f"Features loaded: {X_train.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder


def train_random_forest(X_train, y_train, X_val, y_val):
    """Train Random Forest classifier"""
    logger.info("Training Random Forest...")
    
    start_time = time.time()
    
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=50,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    clf.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    # Evaluate
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    val_acc = accuracy_score(y_val, clf.predict(X_val))
    
    logger.info(f" Random Forest trained in {train_time:.2f}s")
    logger.info(f"   Train Accuracy: {train_acc:.4f}")
    logger.info(f"   Val Accuracy:   {val_acc:.4f}")
    
    return clf, train_time


def train_gradient_boosting(X_train, y_train, X_val, y_val):
    """Train Gradient Boosting classifier"""
    logger.info("Training Gradient Boosting...")
    
    start_time = time.time()
    
    clf = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=10,
        random_state=42,
        verbose=0
    )
    
    clf.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    # Evaluate
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    val_acc = accuracy_score(y_val, clf.predict(X_val))
    
    logger.info(f" Gradient Boosting trained in {train_time:.2f}s")
    logger.info(f"   Train Accuracy: {train_acc:.4f}")
    logger.info(f"   Val Accuracy:   {val_acc:.4f}")
    
    return clf, train_time


def train_naive_bayes(X_train, y_train, X_val, y_val):
    """Train Naive Bayes classifier"""
    logger.info("Training Multinomial Naive Bayes...")
    
    start_time = time.time()
    
    clf = MultinomialNB(alpha=1.0)
    clf.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    # Evaluate
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    val_acc = accuracy_score(y_val, clf.predict(X_val))
    
    logger.info(f" Naive Bayes trained in {train_time:.2f}s")
    logger.info(f"   Train Accuracy: {train_acc:.4f}")
    logger.info(f"   Val Accuracy:   {val_acc:.4f}")
    
    return clf, train_time


def train_svm(X_train, y_train, X_val, y_val):
    """Train Linear SVM classifier"""
    logger.info("Training Linear SVM...")
    
    start_time = time.time()
    
    clf = LinearSVC(
        C=1.0,
        max_iter=1000,
        random_state=42,
        verbose=0
    )
    
    clf.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    # Evaluate
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    val_acc = accuracy_score(y_val, clf.predict(X_val))
    
    logger.info(f" Linear SVM trained in {train_time:.2f}s")
    logger.info(f"   Train Accuracy: {train_acc:.4f}") #hehe fancy
    logger.info(f"   Val Accuracy:   {val_acc:.4f}")
    
    return clf, train_time


def evaluate_model(clf, X_test, y_test, label_encoder, model_name):
    """Comprehensive model evaluation"""
    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUATING: {model_name}")
    logger.info(f"{'='*60}")
    
    # Predictions
    y_pred = clf.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    # ROC-AUC (if model supports predict_proba)
    try:
        y_proba = clf.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        logger.info(f"ROC-AUC: {roc_auc:.4f}")
    except:
        roc_auc = None
        logger.info("ROC-AUC: N/A (model doesn't support probabilities)")
    
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"F1 (macro): {f1_macro:.4f}")
    logger.info(f"F1 (weighted): {f1_weighted:.4f}")
    
    
    
    # Classification report
    report = classification_report(
        y_test, 
        y_pred, 
        target_names=label_encoder.classes_,
        digits=4
    )
    logger.info("\nClassification Report:")
    logger.info("\n" + report)
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'roc_auc': roc_auc
    }


def main():
    logger.info(" Starting Classical ML Models Training")
    logger.info("="*60)
    
    # Load config and features
    config = load_config()
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = load_features()
    
    # Train models
    models = {}
    training_times = {}
    
    # Random Forest
    rf_clf, rf_time = train_random_forest(X_train, y_train, X_val, y_val)
    models['Random Forest'] = rf_clf
    training_times['Random Forest'] = rf_time
    
    # Gradient Boosting
    gb_clf, gb_time = train_gradient_boosting(X_train, y_train, X_val, y_val)
    models['Gradient Boosting'] = gb_clf
    training_times['Gradient Boosting'] = gb_time
    
    # Naive Bayes
    nb_clf, nb_time = train_naive_bayes(X_train, y_train, X_val, y_val)
    models['Naive Bayes'] = nb_clf
    training_times['Naive Bayes'] = nb_time
    
    # Linear SVM
    svm_clf, svm_time = train_svm(X_train, y_train, X_val, y_val)
    models['Linear SVM'] = svm_clf
    training_times['Linear SVM'] = svm_time
    
    # Evaluate all models
    logger.info("\n" + "="*60)
    logger.info("Model Evaluation on Test Set")
    logger.info("="*60)
    
    results = []
    for model_name, clf in models.items():
        result = evaluate_model(clf, X_test, y_test, label_encoder, model_name)
        result['training_time'] = training_times[model_name]
        results.append(result)
    
    # Create comparison table
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('f1_macro', ascending=False)
    
    logger.info("\n" + "="*60)
    logger.info("Model Comparison")
    logger.info("="*60)
    logger.info("\n" + results_df.to_string(index=False))
    
    # Save results
    output_dir = project_root / 'phase2_baseline_models' / 'evaluation'
    output_dir.mkdir(exist_ok=True)
    
    results_df.to_csv(output_dir / 'model_comparison.csv', index=False)
    logger.info(f"\n Results saved to {output_dir / 'model_comparison.csv'}")
    
    # Save best model
    best_model_name = results_df.iloc[0]['model']
    best_model = models[best_model_name]
    
    model_dir = project_root / 'models'
    joblib.dump(best_model, model_dir / f"{best_model_name.lower().replace(' ', '_')}_model.pkl")
    logger.info(f" Best model ({best_model_name}) saved to models/")
    
    logger.info("\n" + "="*60)
    # logger.info(" Done with classical models training and evaluation!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
