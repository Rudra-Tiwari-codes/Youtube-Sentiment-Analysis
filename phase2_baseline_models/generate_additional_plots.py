"""
Generate additional Phase 2 visualizations:
- ROC curves
- Feature importance
- Per-class metrics chart
- Model comparison chart
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score
import joblib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

project_root = Path(__file__).parent.parent
figures_dir = Path(__file__).parent / "figures"
figures_dir.mkdir(exist_ok=True)

print("Loading test data and models...")

# Load test data
tfidf_vectorizer = joblib.load(project_root / "models" / "tfidf_vectorizer.pkl")
label_encoder = joblib.load(project_root / "models" / "label_encoder.pkl")

test_df = pd.read_csv(project_root / "data" / "processed" / "test.csv")
X_test = tfidf_vectorizer.transform(test_df['cleaned_text'])
y_test = label_encoder.transform(test_df['sentiment'])

# Models that support probability predictions
models_with_proba = {
    "Logistic Regression": "logreg_baseline.pkl",
    "Random Forest": "random_forest.pkl",
    "Gradient Boosting": "gradient_boosting.pkl",
    "Naive Bayes": "naive_bayes.pkl"
}

# All models
all_models = {
    **models_with_proba,
    "Linear SVM": "linear_svm.pkl"
}

class_labels = ['Negative', 'Neutral', 'Positive']

# ========================================
# 1. ROC Curves
# ========================================
print("\n1. Generating ROC curves...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
class_names = ['Negative', 'Neutral', 'Positive']

for idx, (model_name, model_file) in enumerate(models_with_proba.items()):
    model_path = project_root / "models" / model_file
    if not model_path.exists():
        print(f"  ⚠️  {model_name} not found, skipping...")
        continue
    
    model = joblib.load(model_path)
    y_proba = model.predict_proba(X_test)
    
    ax = axes[idx]
    
    # Plot ROC for each class (One-vs-Rest)
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        # Binarize the output
        y_binary = (y_test == i).astype(int)
        
        fpr, tpr, _ = roc_curve(y_binary, y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=color, lw=2, 
                label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.3)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / "roc_curves.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: roc_curves.png")
plt.close()

# ========================================
# 2. Feature Importance (TF-IDF)
# ========================================
print("\n2. Generating feature importance plots...")

# Get feature names
feature_names = tfidf_vectorizer.get_feature_names_out()

# For Logistic Regression and Linear SVM
for model_name, model_file in [("Logistic Regression", "logreg_baseline.pkl"), 
                                ("Linear SVM", "linear_svm.pkl")]:
    model_path = project_root / "models" / model_file
    if not model_path.exists():
        continue
    
    model = joblib.load(model_path)
    
    # Get coefficients
    if hasattr(model, 'coef_'):
        coef = model.coef_
        
        # For each class
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (class_name, ax) in enumerate(zip(class_labels, axes)):
            # Top 20 features for this class
            top_indices = np.argsort(np.abs(coef[idx]))[-20:][::-1]
            top_features = feature_names[top_indices]
            top_values = coef[idx][top_indices]
            
            colors = ['green' if v > 0 else 'red' for v in top_values]
            
            ax.barh(range(len(top_features)), top_values, color=colors, alpha=0.7)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features, fontsize=9)
            ax.set_xlabel('Coefficient Value', fontsize=11)
            ax.set_title(f'{class_name} Class', fontsize=12, fontweight='bold')
            ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
            ax.grid(axis='x', alpha=0.3)
        
        plt.suptitle(f'Top 20 Features - {model_name}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        filename = f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(figures_dir / filename, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {filename}")
        plt.close()

# ========================================
# 3. Model Comparison Chart (Interactive HTML)
# ========================================
print("\n3. Generating model comparison chart...")

# Load comparison results
comparison_df = pd.read_csv(project_root / "phase2_baseline_models" / "evaluation" / "model_comparison.csv")

fig = go.Figure()

# Add traces for each metric
metrics = ['accuracy', 'f1_macro', 'f1_weighted']
metric_names = ['Accuracy', 'F1 (Macro)', 'F1 (Weighted)']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for metric, name, color in zip(metrics, metric_names, colors):
    fig.add_trace(go.Bar(
        name=name,
        x=comparison_df['model'],
        y=comparison_df[metric],
        text=[f'{v:.3f}' for v in comparison_df[metric]],
        textposition='outside',
        marker_color=color
    ))

fig.update_layout(
    title='Classical ML Model Comparison',
    xaxis_title='Model',
    yaxis_title='Score',
    barmode='group',
    height=500,
    font=dict(size=12),
    yaxis=dict(range=[0, 1]),
    hovermode='x unified'
)

fig.write_html(figures_dir / "model_comparison.html")
print(f"  ✓ Saved: model_comparison.html")

# ========================================
# 4. Per-Class Performance (Interactive HTML)
# ========================================
print("\n4. Generating per-class metrics chart...")

from sklearn.metrics import precision_recall_fscore_support

# Calculate per-class metrics for all models
per_class_data = []

for model_name, model_file in all_models.items():
    model_path = project_root / "models" / model_file
    if not model_path.exists():
        continue
    
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, 
                                                                 labels=[0, 1, 2],
                                                                 zero_division=0)
    
    for i, class_name in enumerate(class_labels):
        per_class_data.append({
            'Model': model_name,
            'Class': class_name,
            'Precision': precision[i],
            'Recall': recall[i],
            'F1-Score': f1[i]
        })

per_class_df = pd.DataFrame(per_class_data)

# Create grouped bar chart
fig = go.Figure()

for metric in ['Precision', 'Recall', 'F1-Score']:
    for class_name in class_labels:
        data = per_class_df[per_class_df['Class'] == class_name]
        
        fig.add_trace(go.Bar(
            name=f'{class_name} - {metric}',
            x=data['Model'],
            y=data[metric],
            legendgroup=class_name,
            legendgrouptitle_text=class_name,
            hovertemplate='<b>%{x}</b><br>' + metric + ': %{y:.3f}<extra></extra>'
        ))

fig.update_layout(
    title='Per-Class Performance Across Models',
    xaxis_title='Model',
    yaxis_title='Score',
    barmode='group',
    height=600,
    font=dict(size=11),
    yaxis=dict(range=[0, 1]),
    hovermode='closest',
    legend=dict(groupclick="toggleitem")
)

fig.write_html(figures_dir / "per_class_metrics.html")
print(f"  ✓ Saved: per_class_metrics.html")

print(f"\n✓ All Phase 2 visualizations complete!")
print(f"  Location: {figures_dir}")
