"""
Phase 4: Model Comparison Analysis

Comprehensive comparison of all models from Phases 2 and 3.
Visualizes performance improvements and provides recommendations.

Date: December 9, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

project_root = Path(__file__).parent.parent


def collect_model_results():
    """Collect results from all phases"""
    logger.info("Collecting model results...")
    
    models = []
    
    # Phase 2: Classical ML models
    models.append({
        'phase': 'Phase 2',
        'model': 'Logistic Regression',
        'type': 'Classical ML',
        'accuracy': 0.8725,
        'f1_macro': 0.8512,
        'f1_weighted': 0.8698,
        'parameters': '10K features'
    })
    
    models.append({
        'phase': 'Phase 2',
        'model': 'Linear SVM',
        'type': 'Classical ML',
        'accuracy': 0.8857,
        'f1_macro': 0.8663,
        'f1_weighted': 0.8834,
        'parameters': '10K features'
    })
    
    models.append({
        'phase': 'Phase 2',
        'model': 'Gradient Boosting',
        'type': 'Ensemble',
        'accuracy': 0.7543,
        'f1_macro': 0.7125,
        'f1_weighted': 0.7489,
        'parameters': '100 estimators'
    })
    
    models.append({
        'phase': 'Phase 2',
        'model': 'Random Forest',
        'type': 'Ensemble',
        'accuracy': 0.7316,
        'f1_macro': 0.6845,
        'f1_weighted': 0.7254,
        'parameters': '100 estimators'
    })
    
    models.append({
        'phase': 'Phase 2',
        'model': 'Naive Bayes',
        'type': 'Probabilistic',
        'accuracy': 0.7139,
        'f1_macro': 0.6723,
        'f1_weighted': 0.7092,
        'parameters': 'Multinomial'
    })
    
    # Phase 3: Transformer model (ACTUAL TRAINED RESULTS)
    models.append({
        'phase': 'Phase 3',
        'model': 'DistilBERT',
        'type': 'Transformer',
        'accuracy': 0.9647,  # From distilbert_results.txt
        'f1_macro': 0.9563,
        'f1_weighted': 0.9647,
        'parameters': '66M params'
    })
    
    return pd.DataFrame(models)


def create_performance_comparison_plot(df, output_dir):
    """Create comprehensive performance comparison visualization"""
    logger.info("Creating performance comparison plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Sort by accuracy for better visualization
    df_sorted = df.sort_values('accuracy')
    
    # 1. Accuracy comparison
    colors = ['#e74c3c' if phase == 'Phase 2' else '#2ecc71' 
              for phase in df_sorted['phase']]
    
    bars = axes[0, 0].barh(df_sorted['model'], df_sorted['accuracy'] * 100, color=colors, alpha=0.7)
    axes[0, 0].set_xlabel('Accuracy (%)', fontsize=12)
    axes[0, 0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].axvline(x=88.57, color='gray', linestyle='--', label='Phase 2 Best (Linear SVM)', alpha=0.7)
    axes[0, 0].legend()
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        axes[0, 0].text(width, bar.get_y() + bar.get_height()/2,
                       f'{width:.2f}%', ha='left', va='center', fontsize=9)
    
    # 2. F1 Macro comparison
    bars = axes[0, 1].barh(df_sorted['model'], df_sorted['f1_macro'] * 100, color=colors, alpha=0.7)
    axes[0, 1].set_xlabel('F1 Macro (%)', fontsize=12)
    axes[0, 1].set_title('Model F1 Macro Comparison', fontsize=14, fontweight='bold')
    
    for bar in bars:
        width = bar.get_width()
        axes[0, 1].text(width, bar.get_y() + bar.get_height()/2,
                       f'{width:.2f}%', ha='left', va='center', fontsize=9)
    
    # 3. Accuracy vs F1 Weighted scatter
    for phase in df['phase'].unique():
        phase_data = df[df['phase'] == phase]
        color = '#e74c3c' if phase == 'Phase 2' else '#2ecc71'
        axes[1, 0].scatter(phase_data['accuracy'] * 100, phase_data['f1_weighted'] * 100,
                          s=200, alpha=0.6, color=color, label=phase, edgecolors='black')
        
        # Add labels
        for _, row in phase_data.iterrows():
            axes[1, 0].annotate(row['model'], 
                               (row['accuracy'] * 100, row['f1_weighted'] * 100),
                               fontsize=8, ha='right')
    
    axes[1, 0].set_xlabel('Accuracy (%)', fontsize=12)
    axes[1, 0].set_ylabel('F1 Weighted (%)', fontsize=12)
    axes[1, 0].set_title('Accuracy vs F1 Weighted', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Model type comparison (average performance)
    type_performance = df.groupby('type').agg({
        'accuracy': 'mean',
        'f1_macro': 'mean'
    }).reset_index()
    
    x = np.arange(len(type_performance))
    width = 0.35
    
    bars1 = axes[1, 1].bar(x - width/2, type_performance['accuracy'] * 100, 
                          width, label='Accuracy', color='#3498db', alpha=0.7)
    bars2 = axes[1, 1].bar(x + width/2, type_performance['f1_macro'] * 100,
                          width, label='F1 Macro', color='#9b59b6', alpha=0.7)
    
    axes[1, 1].set_xlabel('Model Type', fontsize=12)
    axes[1, 1].set_ylabel('Score (%)', fontsize=12)
    axes[1, 1].set_title('Average Performance by Model Type', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(type_performance['type'])
    axes[1, 1].legend()
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_dir / 'model_comparison.png'}")
    plt.close()


def create_improvement_visualization(df, output_dir):
    """Visualize improvement from Phase 2 to Phase 3"""
    logger.info("Creating improvement visualization...")
    
    # Get best Phase 2 and Phase 3 models
    phase2_best = df[df['phase'] == 'Phase 2'].nlargest(1, 'accuracy').iloc[0]
    phase3_best = df[df['phase'] == 'Phase 3'].nlargest(1, 'accuracy').iloc[0]
    
    metrics = ['Accuracy', 'F1 Macro', 'F1 Weighted']
    phase2_scores = [
        phase2_best['accuracy'] * 100,
        phase2_best['f1_macro'] * 100,
        phase2_best['f1_weighted'] * 100
    ]
    phase3_scores = [
        phase3_best['accuracy'] * 100,
        phase3_best['f1_macro'] * 100,
        phase3_best['f1_weighted'] * 100
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars1 = ax.bar(x - width/2, phase2_scores, width, label=f'Phase 2 ({phase2_best["model"]})',
                  color='#e74c3c', alpha=0.7)
    bars2 = ax.bar(x + width/2, phase3_scores, width, label=f'Phase 3 ({phase3_best["model"]})',
                  color='#2ecc71', alpha=0.7)
    
    ax.set_ylabel('Score (%)', fontsize=13)
    ax.set_title('Performance Improvement: Phase 2 → Phase 3', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)
    
    # Add value labels and improvement arrows
    for i, (bar1, bar2, p2_score, p3_score) in enumerate(zip(bars1, bars2, phase2_scores, phase3_scores)):
        # Phase 2 label
        ax.text(bar1.get_x() + bar1.get_width()/2., p2_score,
               f'{p2_score:.2f}%', ha='center', va='bottom', fontsize=10)
        
        # Phase 3 label
        ax.text(bar2.get_x() + bar2.get_width()/2., p3_score,
               f'{p3_score:.2f}%', ha='center', va='bottom', fontsize=10)
        
        # Improvement arrow and percentage
        improvement = p3_score - p2_score
        ax.annotate('', xy=(bar2.get_x() + bar2.get_width()/2, p3_score - 2),
                   xytext=(bar1.get_x() + bar1.get_width()/2, p2_score + 2),
                   arrowprops=dict(arrowstyle='->', lw=2, color='#34495e'))
        
        mid_y = (p2_score + p3_score) / 2
        ax.text(x[i], mid_y, f'+{improvement:.2f}%',
               ha='center', va='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_improvement.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_dir / 'performance_improvement.png'}")
    plt.close()


def generate_comparison_report(df, output_dir):
    """Generate comprehensive comparison report"""
    logger.info("Generating comparison report...")
    
    report_file = output_dir / 'model_comparison_report.txt'
    
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("PHASE 4: MODEL COMPARISON ANALYSIS\n")
        f.write("="*70 + "\n\n")
        
        # Overall ranking
        f.write("MODEL RANKING (By Accuracy)\n")
        f.write("-"*70 + "\n")
        df_sorted = df.sort_values('accuracy', ascending=False)
        
        for idx, row in df_sorted.iterrows():
            f.write(f"{idx+1}. {row['model']} ({row['phase']})\n")
            f.write(f"   Accuracy: {row['accuracy']:.4f} ({row['accuracy']*100:.2f}%)\n")
            f.write(f"   F1 Macro: {row['f1_macro']:.4f}\n")
            f.write(f"   F1 Weighted: {row['f1_weighted']:.4f}\n")
            f.write(f"   Type: {row['type']} | Params: {row['parameters']}\n\n")
        
        # Phase comparison
        f.write("PHASE-WISE COMPARISON\n")
        f.write("="*70 + "\n\n")
        
        phase2_best = df[df['phase'] == 'Phase 2'].nlargest(1, 'accuracy').iloc[0]
        phase3_best = df[df['phase'] == 'Phase 3'].nlargest(1, 'accuracy').iloc[0]
        
        f.write("PHASE 2 BEST: {}\n".format(phase2_best['model']))
        f.write("-"*70 + "\n")
        f.write(f"  Accuracy: {phase2_best['accuracy']:.4f} ({phase2_best['accuracy']*100:.2f}%)\n")
        f.write(f"  F1 Macro: {phase2_best['f1_macro']:.4f}\n")
        f.write(f"  F1 Weighted: {phase2_best['f1_weighted']:.4f}\n")
        f.write(f"  Type: {phase2_best['type']}\n\n")
        
        f.write("PHASE 3 BEST: {}\n".format(phase3_best['model']))
        f.write("-"*70 + "\n")
        f.write(f"  Accuracy: {phase3_best['accuracy']:.4f} ({phase3_best['accuracy']*100:.2f}%)\n")
        f.write(f"  F1 Macro: {phase3_best['f1_macro']:.4f}\n")
        f.write(f"  F1 Weighted: {phase3_best['f1_weighted']:.4f}\n")
        f.write(f"  Type: {phase3_best['type']}\n\n")
        
        # Improvement analysis
        f.write("IMPROVEMENT ANALYSIS\n")
        f.write("="*70 + "\n\n")
        
        acc_improvement = (phase3_best['accuracy'] - phase2_best['accuracy']) * 100
        f1_improvement = (phase3_best['f1_macro'] - phase2_best['f1_macro']) * 100
        
        f.write(f"Accuracy Improvement: +{acc_improvement:.2f} percentage points\n")
        f.write(f"Relative Improvement: {(acc_improvement/phase2_best['accuracy']):.2f}%\n\n")
        f.write(f"F1 Macro Improvement: +{f1_improvement:.2f} percentage points\n\n")
        
        # Model type analysis
        f.write("MODEL TYPE PERFORMANCE\n")
        f.write("-"*70 + "\n")
        type_perf = df.groupby('type').agg({
            'accuracy': ['mean', 'std', 'max'],
            'f1_macro': ['mean', 'std', 'max']
        }).round(4)
        
        for model_type in df['type'].unique():
            type_data = df[df['type'] == model_type]
            f.write(f"\n{model_type}:\n")
            f.write(f"  Models: {len(type_data)}\n")
            f.write(f"  Avg Accuracy: {type_data['accuracy'].mean():.4f}\n")
            f.write(f"  Best Accuracy: {type_data['accuracy'].max():.4f}\n")
            f.write(f"  Avg F1 Macro: {type_data['f1_macro'].mean():.4f}\n\n")
        
        # Key insights
        f.write("KEY INSIGHTS\n")
        f.write("="*70 + "\n\n")
        
        f.write("1. TRANSFORMER SUPERIORITY:\n")
        f.write("   DistilBERT outperforms all classical ML models by significant margin\n")
        f.write(f"   (+{acc_improvement:.2f}% over best classical model)\n\n")
        
        f.write("2. CLASSICAL ML PERFORMANCE:\n")
        f.write("   - Linear models (SVM, Logistic Reg) perform best\n")
        f.write("   - Tree-based ensembles underperform (likely due to sparse TF-IDF)\n")
        f.write("   - Naive Bayes baseline achieves 71.39% (decent baseline)\n\n")
        
        f.write("3. F1 SCORE CONSISTENCY:\n")
        f.write("   - DistilBERT maintains high F1 across all metrics\n")
        f.write("   - Classical models show more variation between metrics\n\n")
        
        f.write("4. PRODUCTION RECOMMENDATION:\n")
        f.write("   - Primary Model: DistilBERT (96.47% accuracy)\n")
        f.write("   - Fallback Model: Linear SVM (88.57% accuracy, fast inference)\n")
        f.write("   - Use SVM for low-latency requirements, DistilBERT for accuracy\n\n")
        
        f.write("="*70 + "\n")
        f.write("END OF COMPARISON REPORT\n")
        f.write("="*70 + "\n")
    
    logger.info(f"Report saved: {report_file}")


def main():
    logger.info("="*70)
    logger.info("PHASE 4: MODEL COMPARISON ANALYSIS")
    logger.info("="*70)
    
    # Collect results
    df = collect_model_results()
    
    logger.info(f"\nAnalyzing {len(df)} models across 2 phases...")
    
    # Create output directories
    reports_dir = project_root / 'phase4_explainability' / 'reports'
    figures_dir = project_root / 'phase4_explainability' / 'figures'
    reports_dir.mkdir(exist_ok=True, parents=True)
    figures_dir.mkdir(exist_ok=True, parents=True)
    
    # Create visualizations
    create_performance_comparison_plot(df, figures_dir)
    create_improvement_visualization(df, figures_dir)
    
    # Generate report
    generate_comparison_report(df, reports_dir)
    
    # Save comparison CSV
    df.to_csv(reports_dir / 'all_models_comparison.csv', index=False)
    logger.info(f"CSV saved: {reports_dir / 'all_models_comparison.csv'}")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*70)
    
    best_model = df.nlargest(1, 'accuracy').iloc[0]
    logger.info(f"\nBest Model: {best_model['model']}")
    logger.info(f"  Accuracy: {best_model['accuracy']*100:.2f}%")
    logger.info(f"  F1 Macro: {best_model['f1_macro']:.4f}")
    logger.info(f"  Phase: {best_model['phase']}")
    
    logger.info("\n" + "="*70)
    logger.info("OUTPUTS GENERATED:")
    logger.info("="*70)
    logger.info(f"  - {figures_dir / 'model_comparison.png'}")
    logger.info(f"  - {figures_dir / 'performance_improvement.png'}")
    logger.info(f"  - {reports_dir / 'model_comparison_report.txt'}")
    logger.info(f"  - {reports_dir / 'all_models_comparison.csv'}")
    
    logger.info("\n" + "="*70)
    logger.info("MODEL COMPARISON COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
