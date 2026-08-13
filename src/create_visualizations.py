"""
VISUALIZATION & PLOTS: Create All Diagrams for Research Paper
==============================================================

Creates publication-ready visualizations:
1. Confusion Matrix (heatmap)
2. ROC Curve
3. Feature Importance (bar plot)
4. Cross-Validation Scores
5. Model Comparison
6. Class Distribution
7. Feature Distribution
8. Precision-Recall Curve

Run with: python create_visualizations.py

Author: [Your Name]
Date: [Current Date]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report, accuracy_score
)
import pickle
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# FIGURE 1: CONFUSION MATRIX HEATMAP
# ============================================================================

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Create confusion matrix heatmap
    
    Shows:
    - True Positives (TP): Correctly predicted hypothyroid
    - True Negatives (TN): Correctly predicted negative
    - False Positives (FP): Incorrectly predicted hypothyroid
    - False Negatives (FN): Incorrectly predicted negative
    """
    
    print("\n📊 Creating Confusion Matrix...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                cbar_kws={'label': 'Count'},
                xticklabels=['Negative', 'Hypothyroid'],
                yticklabels=['Negative', 'Hypothyroid'],
                annot_kws={'size': 14, 'weight': 'bold'},
                linewidths=2, linecolor='black')
    
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add accuracy annotation
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    plt.text(1, -0.2, f'Accuracy: {accuracy:.2%}', 
             ha='center', fontsize=11, fontweight='bold',
             transform=plt.gca().transAxes)
    
    plt.tight_layout()
    
    # Save figure
    filepath = f"{OUTPUT_DIR}/01_confusion_matrix.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    plt.close()
    
    # Print matrix details
    print(f"\n  Confusion Matrix Details:")
    print(f"  ├─ True Negatives (TN):      {cm[0,0]:5d}")
    print(f"  ├─ False Positives (FP):     {cm[0,1]:5d}")
    print(f"  ├─ False Negatives (FN):     {cm[1,0]:5d}")
    print(f"  └─ True Positives (TP):      {cm[1,1]:5d}")
    print(f"\n  Calculations:")
    print(f"  ├─ Sensitivity (Recall):     {cm[1,1]/(cm[1,0]+cm[1,1]):.4f}")
    print(f"  ├─ Specificity:              {cm[0,0]/(cm[0,0]+cm[0,1]):.4f}")
    print(f"  └─ Accuracy:                 {accuracy:.4f}")

# ============================================================================
# FIGURE 2: ROC CURVE
# ============================================================================

def plot_roc_curve(y_true, y_proba, title="ROC Curve"):
    """
    Create ROC (Receiver Operating Characteristic) Curve
    
    Shows:
    - Relationship between True Positive Rate (TPR) and False Positive Rate (FPR)
    - AUC (Area Under Curve): Probability that model ranks positive higher than negative
    - Diagonal line: Random classifier (50% accuracy)
    """
    
    print("\n📊 Creating ROC Curve...")
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2.5, 
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
    
    # Plot diagonal (random classifier)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier (AUC = 0.5)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    filepath = f"{OUTPUT_DIR}/02_roc_curve.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    plt.close()
    
    print(f"\n  ROC Curve Interpretation:")
    print(f"  ├─ AUC = {roc_auc:.4f} (Area Under Curve)")
    print(f"  ├─ AUC > 0.9: Excellent discrimination")
    print(f"  ├─ AUC = 0.5: Random classifier")
    print(f"  └─ Your model: {'✓ EXCELLENT' if roc_auc > 0.95 else 'Good' if roc_auc > 0.90 else 'Fair'}")

# ============================================================================
# FIGURE 3: FEATURE IMPORTANCE BAR PLOT
# ============================================================================

def plot_feature_importance(importance_df, top_n=15, title="Feature Importance"):
    """
    Create feature importance bar plot
    
    Shows:
    - Which features contribute most to predictions
    - Percentage contribution of each feature
    - Ranks features by importance
    """
    
    print("\n📊 Creating Feature Importance Plot...")
    
    # Get top N features
    top_features = importance_df.head(top_n)
    
    # Highlight new features
    colors = ['#2ecc71' if feat in ['family_history_level', 'weight_change_kg', 'weight_change_norm'] 
              else '#3498db' for feat in top_features['Feature']]
    
    plt.figure(figsize=(10, 8))
    
    # Create bar plot
    bars = plt.barh(range(len(top_features)), top_features['Percent'], color=colors)
    
    # Add value labels
    for i, (idx, row) in enumerate(top_features.iterrows()):
        plt.text(row['Percent'] + 0.5, i, f"{row['Percent']:.2f}%", 
                va='center', fontweight='bold', fontsize=9)
    
    plt.yticks(range(len(top_features)), top_features['Feature'], fontsize=10)
    plt.xlabel('Importance (%)', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlim([0, top_features['Percent'].max() + 5])
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Novel Features (Your Contribution)'),
        Patch(facecolor='#3498db', label='Original Features')
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    filepath = f"{OUTPUT_DIR}/03_feature_importance.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    plt.close()
    
    # Print feature importance details
    print(f"\n  Top {top_n} Features:")
    for i, (idx, row) in enumerate(top_features.iterrows(), 1):
        marker = "✓ NEW" if row['Feature'] in ['family_history_level', 'weight_change_kg', 'weight_change_norm'] else ""
        print(f"  {i:2d}. {row['Feature']:30s} {row['Percent']:6.2f}% {marker}")

# ============================================================================
# FIGURE 4: CROSS-VALIDATION SCORES
# ============================================================================

def plot_cross_validation_scores(cv_scores, title="5-Fold Cross-Validation Scores"):
    """
    Create cross-validation scores visualization
    
    Shows:
    - Accuracy for each fold
    - Mean and standard deviation
    - Stability of model performance
    """
    
    print("\n📊 Creating Cross-Validation Plot...")
    
    folds = [f'Fold {i+1}' for i in range(len(cv_scores))]
    
    plt.figure(figsize=(10, 6))
    
    # Plot scores
    bars = plt.bar(folds, cv_scores * 100, color='#3498db', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, score in enumerate(cv_scores):
        plt.text(i, score * 100 + 0.1, f'{score*100:.2f}%', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Add mean line
    mean_score = cv_scores.mean()
    plt.axhline(y=mean_score * 100, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_score*100:.2f}%')
    
    # Add std band
    std_score = cv_scores.std()
    plt.fill_between(range(len(cv_scores)), 
                     (mean_score - std_score) * 100,
                     (mean_score + std_score) * 100,
                     alpha=0.2, color='red',
                     label=f'±1 Std: {std_score*100:.2f}%')
    
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylim([98, 100])
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    filepath = f"{OUTPUT_DIR}/04_cross_validation.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    plt.close()
    
    print(f"\n  Cross-Validation Statistics:")
    print(f"  ├─ Mean Accuracy:     {mean_score*100:.2f}%")
    print(f"  ├─ Std Deviation:     {std_score*100:.2f}%")
    print(f"  ├─ Min Accuracy:      {cv_scores.min()*100:.2f}%")
    print(f"  ├─ Max Accuracy:      {cv_scores.max()*100:.2f}%")
    print(f"  └─ Stability:         {'✓ EXCELLENT (std < 0.01)' if std_score < 0.01 else 'Good' if std_score < 0.02 else 'Fair'}")

# ============================================================================
# FIGURE 5: CLASS DISTRIBUTION
# ============================================================================

def plot_class_distribution(y_true, title="Class Distribution"):
    """
    Create class distribution pie chart
    
    Shows:
    - Balance between hypothyroid and negative samples
    - Percentage of each class
    """
    
    print("\n📊 Creating Class Distribution Plot...")
    
    classes = ['Negative', 'Hypothyroid']
    counts = pd.Series(y_true).value_counts().sort_index()
    colors = ['#3498db', '#e74c3c']
    
    plt.figure(figsize=(8, 6))
    
    wedges, texts, autotexts = plt.pie(counts, labels=classes, autopct='%1.1f%%',
                                         colors=colors, startangle=90,
                                         textprops={'fontsize': 12, 'weight': 'bold'})
    
    # Add count labels
    for i, (count, autotext) in enumerate(zip(counts, autotexts)):
        texts[i].set_text(f'{classes[i]}\n(n={count})')
        texts[i].set_fontsize(11)
        texts[i].set_weight('bold')
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save figure
    filepath = f"{OUTPUT_DIR}/05_class_distribution.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    plt.close()
    
    print(f"\n  Class Distribution:")
    for i, (cls, count) in enumerate(zip(classes, counts)):
        print(f"  ├─ {cls:20s}: {count:5d} ({count/sum(counts)*100:5.1f}%)")

# ============================================================================
# FIGURE 6: PRECISION-RECALL CURVE
# ============================================================================

def plot_precision_recall_curve(y_true, y_proba, title="Precision-Recall Curve"):
    """
    Create Precision-Recall Curve
    
    Shows:
    - Trade-off between precision and recall
    - Better for imbalanced datasets than ROC
    """
    
    print("\n📊 Creating Precision-Recall Curve...")
    
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    
    # Plot precision-recall curve
    plt.plot(recall, precision, color='green', lw=2.5, 
            label=f'Precision-Recall Curve (AUC = {pr_auc:.4f})')
    
    # Plot baseline
    no_skill = sum(y_true) / len(y_true)
    plt.axhline(y=no_skill, color='gray', linestyle='--', lw=2,
               label=f'Baseline ({no_skill:.2%})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="upper right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    filepath = f"{OUTPUT_DIR}/06_precision_recall_curve.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    plt.close()

# ============================================================================
# FIGURE 7: MODEL METRICS COMPARISON
# ============================================================================

def plot_model_metrics(metrics_dict, title="Model Performance Metrics"):
    """
    Create model metrics bar chart
    
    Shows:
    - Accuracy, Precision, Recall, F1, ROC-AUC
    - Quick comparison of all metrics
    """
    
    print("\n📊 Creating Model Metrics Plot...")
    
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    metrics_values = [
        metrics_dict.get('test_acc', 0),
        metrics_dict.get('precision', 0),
        metrics_dict.get('recall', 0),
        metrics_dict.get('f1', 0),
        metrics_dict.get('auc', 0)
    ]
    
    plt.figure(figsize=(10, 6))
    
    # Create bar plot
    bars = plt.bar(metrics_names, metrics_values, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6'],
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}\n({value*100:.2f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Add threshold lines
    plt.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='Excellent (0.95)')
    plt.axhline(y=0.90, color='orange', linestyle='--', alpha=0.5, label='Good (0.90)')
    
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylim([0.85, 1.05])
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    filepath = f"{OUTPUT_DIR}/07_metrics_comparison.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    plt.close()
    
    print(f"\n  Model Metrics:")
    for name, value in zip(metrics_names, metrics_values):
        print(f"  ├─ {name:15s}: {value:.4f} ({value*100:.2f}%)")

# ============================================================================
# FIGURE 8: FEATURE DISTRIBUTION BY CLASS
# ============================================================================

def plot_feature_distribution(df, feature_cols, title="Feature Distributions by Class"):
    """
    Create feature distribution plots by class
    
    Shows:
    - Distribution of key features
    - Separation between hypothyroid and healthy
    """
    
    print("\n📊 Creating Feature Distribution Plot...")
    
    # Select top features
    features_to_plot = ['family_history_level', 'weight_change_kg', 'TSH']
    features_to_plot = [f for f in features_to_plot if f in feature_cols]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, feature in enumerate(features_to_plot):
        ax = axes[idx]
        
        # Get data by class
        hypothyroid_data = df[df['class'] == 'hypothyroid'][feature]
        healthy_data = df[df['class'] == 'negative'][feature]
        
        # Plot distributions
        ax.hist(hypothyroid_data, bins=30, alpha=0.6, label='Hypothyroid', color='#e74c3c', edgecolor='black')
        ax.hist(healthy_data, bins=30, alpha=0.6, label='Healthy', color='#3498db', edgecolor='black')
        
        ax.set_xlabel(feature, fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    filepath = f"{OUTPUT_DIR}/08_feature_distributions.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Generate all visualizations"""
    
    print("\n" + "=" * 80)
    print("CREATING PUBLICATION-READY VISUALIZATIONS")
    print("=" * 80)
    
    # For demonstration, create sample data
    # In real usage, load actual model predictions and data
    
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    
    # Create sample data
    X, y = make_classification(n_samples=3865, n_features=25, n_informative=15,
                               n_redundant=5, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    metrics = {
        'test_acc': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba)
    }
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    # Feature importance
    importance_df = pd.DataFrame({
        'Feature': [f'Feature_{i}' for i in range(X.shape[1])],
        'Importance': model.feature_importances_,
        'Percent': model.feature_importances_ * 100
    }).sort_values('Importance', ascending=False)
    
    # Create visualizations
    print("\n✓ Generating visualizations...")
    
    plot_confusion_matrix(y_test, y_pred, "Confusion Matrix - Test Set")
    plot_roc_curve(y_test, y_proba, "ROC Curve (Test Set)")
    plot_feature_importance(importance_df, top_n=15, title="Feature Importance Analysis")
    plot_cross_validation_scores(cv_scores, "5-Fold Cross-Validation Results")
    plot_class_distribution(y_test, "Dataset Class Distribution")
    plot_precision_recall_curve(y_test, y_proba, "Precision-Recall Curve")
    plot_model_metrics(metrics, "Model Performance Metrics")
    
    # For feature distribution, create sample DataFrame
    df = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(X.shape[1])])
    df['class'] = np.where(y == 1, 'hypothyroid', 'negative')
    
    # Add sample columns for demonstration
    df['family_history_level'] = np.random.randint(0, 4, size=len(df))
    df['weight_change_kg'] = np.random.normal(1.5, 1.5, size=len(df))
    df['TSH'] = np.random.normal(5, 3, size=len(df))
    
    plot_feature_distribution(df, df.columns, "Feature Distributions by Disease Status")
    
    # Final summary
    print("\n" + "=" * 80)
    print("VISUALIZATIONS COMPLETE!")
    print("=" * 80)
    
    print(f"\n✓ All figures saved to: {OUTPUT_DIR}/")
    print(f"\nGenerated Figures:")
    print(f"  1. 01_confusion_matrix.png - Confusion Matrix Heatmap")
    print(f"  2. 02_roc_curve.png - ROC Curve")
    print(f"  3. 03_feature_importance.png - Feature Importance Bar Plot")
    print(f"  4. 04_cross_validation.png - Cross-Validation Scores")
    print(f"  5. 05_class_distribution.png - Class Distribution Pie Chart")
    print(f"  6. 06_precision_recall_curve.png - Precision-Recall Curve")
    print(f"  7. 07_metrics_comparison.png - Model Metrics Comparison")
    print(f"  8. 08_feature_distributions.png - Feature Distributions")
    
    print(f"\n✓ Use these figures in your research paper!")
    print(f"✓ All figures are high-resolution (300 DPI) and publication-ready")
    print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    main()
