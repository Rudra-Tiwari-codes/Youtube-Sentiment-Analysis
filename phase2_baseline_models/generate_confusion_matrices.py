"""
Generate confusion matrix visualizations for all trained models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
from pathlib import Path

project_root = Path(__file__).parent.parent
figures_dir = Path(__file__).parent / "figures"
figures_dir.mkdir(exist_ok=True)

print("Loading test data and models...")

# Load test features and labels from baseline output
tfidf_vectorizer = joblib.load(project_root / "models" / "tfidf_vectorizer.pkl")
label_encoder = joblib.load(project_root / "models" / "label_encoder.pkl")

# Load original test data
test_df = pd.read_csv(project_root / "data" / "processed" / "test.csv")
X_test = tfidf_vectorizer.transform(test_df['cleaned_text'])
y_test = label_encoder.transform(test_df['sentiment'])

# Model names and files
models = {
    "Random Forest": "random_forest.pkl",
    "Gradient Boosting": "gradient_boosting.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Linear SVM": "linear_svm.pkl"
}

# Class labels
class_labels = ['Negative', 'Neutral', 'Positive']

print("\nGenerating confusion matrices...\n")

for model_name, model_file in models.items():
    model_path = project_root / "models" / model_file
    
    if not model_path.exists():
        print(f"[WARNING] {model_name} model not found, skipping...")
        continue
    
    # Load model
    model = joblib.load(model_path)
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save figure
    output_file = figures_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file.name}")
    plt.close()

print(f"\n✓ All confusion matrices saved to: {figures_dir}")
