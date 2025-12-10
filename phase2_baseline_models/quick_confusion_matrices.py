"""
Quick confusion matrix generator for available models
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
from pathlib import Path

project_root = Path(__file__).parent.parent
figures_dir = Path(__file__).parent / "figures"
figures_dir.mkdir(exist_ok=True)

print("Loading test data...")

# Load test data and preprocessing
tfidf_vectorizer = joblib.load(project_root / "models" / "tfidf_vectorizer.pkl")
label_encoder = joblib.load(project_root / "models" / "label_encoder.pkl")

test_df = pd.read_csv(project_root / "data" / "processed" / "test.csv")
X_test = tfidf_vectorizer.transform(test_df['cleaned_text'])
y_test = label_encoder.transform(test_df['sentiment'])

# Class labels
class_labels = ['Negative', 'Neutral', 'Positive']

# Find all model files
model_dir = project_root / "models"
model_files = list(model_dir.glob("*.pkl"))

print(f"\nFound {len(model_files)} files in models/")
print("\nGenerating confusion matrices...\n")

generated_count = 0

for model_path in model_files:
    # Skip non-model files
    if model_path.name in ['label_encoder.pkl', 'tfidf_vectorizer.pkl']:
        continue
    
    try:
        # Load model
        model = joblib.load(model_path)
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_labels, yticklabels=class_labels,
                    cbar_kws={'label': 'Count'}, annot_kws={'size': 14})
        
        # Clean model name
        model_name = model_path.stem.replace('_', ' ').replace('model', '').strip().title()
        if not model_name:
            model_name = model_path.stem
        
        plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=13)
        plt.xlabel('Predicted Label', fontsize=13)
        plt.tight_layout()
        
        # Save figure
        output_file = figures_dir / f"confusion_matrix_{model_path.stem}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file.name}")
        plt.close()
        
        generated_count += 1
        
    except Exception as e:
        print(f"⚠️  Skipped {model_path.name}: {str(e)}")

print(f"\n✓ Generated {generated_count} confusion matrices in: {figures_dir}")
