"""
                    Advanced Error Analysis
Comprehensive error analysis for DistilBERT sentiment classifier.
Identifies patterns in misclassifications and model weaknesses.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

project_root = Path(__file__).parent.parent


def load_results():
    """Load DistilBERT results from Colab training"""
    logger.info("Loading test results...")
    
    # Load test data
    data_dir = project_root / 'data' / 'processed'
    test_df = pd.read_csv(data_dir / 'test.csv')
    
    # Load label encoder
    label_encoder = joblib.load(project_root / 'models' / 'label_encoder.pkl')
    
    # Parse results from distilbert_results.txt
    results_file = project_root / 'distilbert_results.txt'
    
    logger.info(f"Test set: {len(test_df):,} samples")
    logger.info(f"Classes: {label_encoder.classes_}")
    
    return test_df, label_encoder


def create_confusion_matrix_plot(y_true, y_pred, class_names, output_dir):
    """Create enhanced confusion matrix visualization"""
    logger.info("Creating confusion matrix visualization...")
    
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Absolute counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    
    # Normalized percentages
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={'label': 'Proportion'})
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_dir / 'confusion_matrix.png'}")
    plt.close()
    
    return cm, cm_normalized


def analyze_per_class_errors(y_true, y_pred, class_names):
    """Detailed per-class error analysis"""
    logger.info("Analyzing per-class errors...")
    
    results = []
    
    for i, class_name in enumerate(class_names):
        # Get samples for this class
        class_mask = y_true == i
        class_true = y_true[class_mask]
        class_pred = y_pred[class_mask]
        
        # Calculate metrics
        total = len(class_true)
        correct = np.sum(class_pred == i)
        accuracy = correct / total if total > 0 else 0
        
        # Misclassification breakdown
        misclass_counts = {}
        for j, target_class in enumerate(class_names):
            if j != i:
                count = np.sum(class_pred == j)
                misclass_counts[target_class] = count
        
        results.append({
            'class': class_name,
            'total_samples': total,
            'correct': correct,
            'errors': total - correct,
            'accuracy': accuracy,
            'error_rate': 1 - accuracy,
            'misclassifications': misclass_counts
        })
    
    return results



# Optional for non-tech people to see  (used AI to help write this)

def create_error_analysis_report(error_analysis, cm, cm_normalized, output_dir):
    """Generate error analysis report"""
    logger.info("Generating error analysis report...")
    
    report_file = output_dir / 'error_analysis_report.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("Error Analysis Report\n")
        f.write("DistilBERT Sentiment Classifier\n")
        f.write("="*70 + "\n\n")
        f.write("-"*70 + "\n")
        total_samples = sum(result['total_samples'] for result in error_analysis)
        total_errors = sum(result['errors'] for result in error_analysis)
        overall_accuracy = 1 - (total_errors / total_samples)
        
        f.write(f"Total Test Samples: {total_samples:,}\n")
        f.write(f"Total Correct: {total_samples - total_errors:,}\n")
        f.write(f"Total Errors: {total_errors:,}\n")
        f.write(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)\n")
        f.write(f"Overall Error Rate: {total_errors/total_samples:.4f} ({(total_errors/total_samples)*100:.2f}%)\n\n")
        
        # Per-class analysis
        f.write("PER-CLASS ERROR ANALYSIS\n")
        f.write("="*70 + "\n\n")
        
        for result in error_analysis:
            f.write(f"CLASS: {result['class'].upper()}\n")
            f.write("-"*70 + "\n")
            f.write(f"  Total Samples: {result['total_samples']:,}\n")
            f.write(f"  Correct Predictions: {result['correct']:,}\n")
            f.write(f"  Misclassifications: {result['errors']:,}\n")
            f.write(f"  Class Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)\n")
            f.write(f"  Class Error Rate: {result['error_rate']:.4f} ({result['error_rate']*100:.2f}%)\n")
            f.write(f"\n  Misclassification Breakdown:\n")
            
            for target_class, count in result['misclassifications'].items():
                if count > 0:
                    pct = (count / result['total_samples']) * 100
                    f.write(f"    → Predicted as {target_class}: {count:,} ({pct:.2f}%)\n")
            f.write("\n")
        
        # Confusion matrix
        f.write("CONFUSION MATRIX (COUNTS)\n")
        f.write("-"*70 + "\n")
        f.write(str(cm) + "\n\n")
        
        f.write("CONFUSION MATRIX (NORMALIZED)\n")
        f.write("-"*70 + "\n")
        f.write(str(cm_normalized) + "\n\n")
        
        # Error patterns
        f.write("ERROR PATTERN ANALYSIS\n")
        f.write("="*70 + "\n\n")
        
        # Most confused pairs
        max_confusion = 0
        confused_pair = None
        
        for i in range(len(cm)):
            for j in range(len(cm)):
                if i != j and cm[i][j] > max_confusion:
                    max_confusion = cm[i][j]
                    confused_pair = (error_analysis[i]['class'], error_analysis[j]['class'])
        
        if confused_pair:
            f.write(f"Most Confused Class Pair:\n")
            f.write(f"  {confused_pair[0]} ↔ {confused_pair[1]}: {max_confusion} misclassifications\n\n")
        
        # Hardest class to predict
        hardest_class = max(error_analysis, key=lambda x: x['error_rate'])
        f.write(f"Most Challenging Class:\n")
        f.write(f"  {hardest_class['class']}: {hardest_class['error_rate']*100:.2f}% error rate\n\n")
        
        # Best performing class
        best_class = min(error_analysis, key=lambda x: x['error_rate'])
        f.write(f"Best Performing Class:\n")
        f.write(f"  {best_class['class']}: {best_class['accuracy']*100:.2f}% accuracy\n\n")
        
        f.write("="*70 + "\n")
        f.write("END OF ERROR ANALYSIS\n")
        f.write("="*70 + "\n")
    
    logger.info(f"Report saved: {report_file}")


def create_class_performance_plot(error_analysis, output_dir):
    """Create bar plot of per-class performance"""
    logger.info("Creating class performance visualization...")
    
    classes = [r['class'] for r in error_analysis]
    accuracies = [r['accuracy'] * 100 for r in error_analysis]
    error_rates = [r['error_rate'] * 100 for r in error_analysis]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy by class
    bars1 = axes[0].bar(classes, accuracies, color=['#2ecc71', '#3498db', '#e74c3c'])
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 100)
    axes[0].axhline(y=96.47, color='gray', linestyle='--', label='Overall Accuracy', alpha=0.7)
    axes[0].legend()
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%', ha='center', va='bottom', fontsize=10)
    
    # Error rate by class
    bars2 = axes[1].bar(classes, error_rates, color=['#e74c3c', '#3498db', '#2ecc71'])
    axes[1].set_ylabel('Error Rate (%)', fontsize=12)
    axes[1].set_title('Per-Class Error Rate', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, max(error_rates) * 1.3)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_performance.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_dir / 'class_performance.png'}")
    plt.close()


def simulate_predictions_from_results():
    """
    Simulate predictions from confusion matrix in distilbert_results.txt
    This allows us to perform error analysis even though we don't have the actual predictions
    """
    logger.info("Simulating predictions from results file...")
    
    # Confusion matrix from results
    # [[1932   37  119]
    #  [  63 5552  106]
    #  [  96   44 5212]]
    
    # Class distribution from results
    # Negative: 2088, Neutral: 5721, Positive: 5352
    
    y_true = []
    y_pred = []
    
    # Negative class (label 0)
    y_true.extend([0] * 1932); y_pred.extend([0] * 1932)  # Correct
    y_true.extend([0] * 37); y_pred.extend([1] * 37)      # Predicted as Neutral
    y_true.extend([0] * 119); y_pred.extend([2] * 119)    # Predicted as Positive
    
    # Neutral class (label 1)
    y_true.extend([1] * 63); y_pred.extend([0] * 63)      # Predicted as Negative
    y_true.extend([1] * 5552); y_pred.extend([1] * 5552)  # Correct
    y_true.extend([1] * 106); y_pred.extend([2] * 106)    # Predicted as Positive
    
    # Positive class (label 2)
    y_true.extend([2] * 96); y_pred.extend([0] * 96)      # Predicted as Negative
    y_true.extend([2] * 44); y_pred.extend([1] * 44)      # Predicted as Neutral
    y_true.extend([2] * 5212); y_pred.extend([2] * 5212)  # Correct
    
    return np.array(y_true), np.array(y_pred)


def main():
    logger.info("="*70)
    logger.info("PHASE 4: ERROR ANALYSIS")
    logger.info("="*70)
    
    # Load data
    test_df, label_encoder = load_results()
    class_names = label_encoder.classes_
    
    # Simulate predictions from confusion matrix
    y_true, y_pred = simulate_predictions_from_results()
    
    logger.info(f"\nAnalyzing {len(y_true):,} predictions...")
    logger.info(f"Classes: {class_names}")
    
    # Create output directory
    output_dir = project_root / 'phase4_explainability' / 'reports'
    figures_dir = project_root / 'phase4_explainability' / 'figures'
    output_dir.mkdir(exist_ok=True, parents=True)
    figures_dir.mkdir(exist_ok=True, parents=True)
    
    # Confusion matrix visualization
    cm, cm_normalized = create_confusion_matrix_plot(
        y_true, y_pred, class_names, figures_dir
    )
    
    # Per-class error analysis
    error_analysis = analyze_per_class_errors(y_true, y_pred, class_names)
    
    # Class performance plots
    create_class_performance_plot(error_analysis, figures_dir)
    
    # Generate comprehensive report
    create_error_analysis_report(error_analysis, cm, cm_normalized, output_dir)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("ERROR ANALYSIS SUMMARY")
    logger.info("="*70)
    
    for result in error_analysis:
        logger.info(f"\n{result['class']}:")
        logger.info(f"  Accuracy: {result['accuracy']*100:.2f}%")
        logger.info(f"  Errors: {result['errors']:,} / {result['total_samples']:,}")
    
    logger.info("\n" + "="*70)
    logger.info("Outputs png and txt files  ")
    logger.info("="*70)
    logger.info(f"  - {figures_dir / 'confusion_matrix.png'}")
    logger.info(f"  - {figures_dir / 'class_performance.png'}")
    logger.info(f"  - {output_dir / 'error_analysis_report.txt'}")
    
    logger.info("\n" + "="*70)
    logger.info("ERROR ANALYSIS COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
