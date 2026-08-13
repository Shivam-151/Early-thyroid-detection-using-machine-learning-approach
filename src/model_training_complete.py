"""
MODEL TRAINING: ENSEMBLE WITH VALIDATION
========================================

This script trains an optimized ensemble model for thyroid disease prediction:
1. Weighted ensemble: Random Forest (50%), SVM (30%), KNN (20%)
2. 5-fold cross-validation for stability assessment
3. Feature importance analysis
4. Model evaluation with multiple metrics

Author: [Your Name]
Date: [Current Date]
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

# ============================================================================
# STEP 1: LOAD ENHANCED DATASET
# ============================================================================

def load_data(filepath):
    """
    Load enhanced dataset with new features
    
    Parameters:
    -----------
    filepath : str
        Path to enhanced dataset
    
    Returns:
    --------
    DataFrame : Loaded dataset
    """
    
    print("=" * 80)
    print("STEP 1: LOADING ENHANCED DATASET")
    print("=" * 80)
    
    df = pd.read_csv(filepath)
    
    print(f"\n✓ Dataset loaded successfully!")
    print(f"  - Total patients: {len(df)}")
    print(f"  - Total features: {len(df.columns)}")
    print(f"  - Target variable: 'class' (hypothyroid/negative)")
    print(f"\nDataset shape: {df.shape}")
    
    # Show class distribution
    print("\nClass distribution:")
    print(df['class'].value_counts())
    print(f"\nClass proportions:")
    print(df['class'].value_counts(normalize=True))
    
    return df

# ============================================================================
# STEP 2: PREPARE DATA FOR MODELING
# ============================================================================

def prepare_data(df):
    """
    Prepare data: encode target, separate features/target, handle NaN
    
    Parameters:
    -----------
    df : DataFrame
        Dataset with features and target
    
    Returns:
    --------
    X : DataFrame - Features
    y : Series - Target variable
    label_encoder : LabelEncoder - For encoding disease status
    feature_names : list - Names of all features
    """
    
    print("\n" + "=" * 80)
    print("STEP 2: PREPARING DATA")
    print("=" * 80)
    
    # Handle missing values
    print("\n✓ Handling missing values...")
    print(f"  - Missing values before: {df.isnull().sum().sum()}")
    
    df = df.fillna(df.mean(numeric_only=True))
    print(f"  - Missing values after: {df.isnull().sum().sum()}")
    
    # Separate features and target
    X = df.drop('class', axis=1)
    y = df['class']
    
    # Encode target variable
    print("\n✓ Encoding target variable...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"  - Classes: {label_encoder.classes_}")
    print(f"  - Encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    feature_names = X.columns.tolist()
    
    print(f"\n✓ Data preparation complete!")
    print(f"  - Features shape: {X.shape}")
    print(f"  - Target shape: {y_encoded.shape}")
    
    return X, y_encoded, label_encoder, feature_names

# ============================================================================
# STEP 3: SPLIT DATA
# ============================================================================

def split_data(X, y):
    """
    Split data into train, validation, and test sets
    
    Parameters:
    -----------
    X : DataFrame - Features
    y : array - Target
    
    Returns:
    --------
    X_train, X_test, y_train, y_test : Train/test splits
    """
    
    print("\n" + "=" * 80)
    print("STEP 3: SPLITTING DATA")
    print("=" * 80)
    
    # 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # Maintain class distribution
    )
    
    print(f"\n✓ Data split successfully!")
    print(f"  - Train set: {X_train.shape[0]} patients ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"  - Test set: {X_test.shape[0]} patients ({X_test.shape[0]/len(X)*100:.1f}%)")
    print(f"\n  Train set class distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"    Class {cls}: {count} ({count/len(y_train)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test

# ============================================================================
# STEP 4: FEATURE SCALING
# ============================================================================

def scale_features(X_train, X_test):
    """
    Scale features to standardized values (mean=0, std=1)
    
    Important for SVM and KNN algorithms
    
    Parameters:
    -----------
    X_train : DataFrame - Training features
    X_test : DataFrame - Test features
    
    Returns:
    --------
    X_train_scaled, X_test_scaled : Scaled features
    scaler : StandardScaler object (for deployment)
    """
    
    print("\n" + "=" * 80)
    print("STEP 4: SCALING FEATURES")
    print("=" * 80)
    
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for easier handling
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    print(f"\n✓ Features scaled successfully!")
    print(f"  - Scaling method: StandardScaler (mean=0, std=1)")
    print(f"  - Train set - Mean: {X_train_scaled.mean().mean():.6f}, Std: {X_train_scaled.std().mean():.6f}")
    print(f"  - Test set - Mean: {X_test_scaled.mean().mean():.6f}, Std: {X_test_scaled.std().mean():.6f}")
    
    return X_train_scaled, X_test_scaled, scaler

# ============================================================================
# STEP 5: BUILD ENSEMBLE MODEL
# ============================================================================

def build_ensemble_model():
    """
    Build weighted ensemble classifier
    
    Components:
    - Random Forest (50% weight): Good for feature interactions
    - SVM (30% weight): Good for non-linear patterns
    - KNN (20% weight): Good for local patterns
    
    Soft voting: Uses probability predictions from each classifier
    
    Returns:
    --------
    VotingClassifier : Ensemble model ready to train
    """
    
    print("\n" + "=" * 80)
    print("STEP 5: BUILDING ENSEMBLE MODEL")
    print("=" * 80)
    
    # Individual classifiers
    print("\n✓ Building individual classifiers...")
    
    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,      # 100 trees
        max_depth=15,          # Tree depth
        min_samples_split=5,   # Min samples to split node
        min_samples_leaf=2,    # Min samples in leaf
        random_state=42,
        n_jobs=-1              # Use all processors
    )
    print("  - Random Forest (100 trees, max_depth=15)")
    
    # SVM with probability
    svm = SVC(
        kernel='rbf',          # Radial basis function
        gamma='scale',
        probability=True,      # Enable probability output
        random_state=42
    )
    print("  - SVM (RBF kernel)")
    
    # KNN
    knn = KNeighborsClassifier(
        n_neighbors=5,         # 5 nearest neighbors
        weights='distance',    # Weight by distance
        n_jobs=-1
    )
    print("  - KNN (n_neighbors=5, distance weights)")
    
    # Voting Classifier
    print("\n✓ Creating weighted ensemble...")
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('svm', svm),
            ('knn', knn)
        ],
        voting='soft',         # Soft voting (uses probabilities)
        weights=[0.5, 0.3, 0.2]  # Optimized weights
    )
    
    print("\nEnsemble configuration:")
    print("  - Voting method: Soft (probability-based)")
    print("  - Weights: RF=0.5, SVM=0.3, KNN=0.2")
    print("  - Final prediction: Weighted average of probabilities")
    
    return ensemble

# ============================================================================
# STEP 6: TRAIN MODEL
# ============================================================================

def train_model(ensemble, X_train, y_train):
    """
    Train ensemble model
    
    Parameters:
    -----------
    ensemble : VotingClassifier - Ensemble model
    X_train : DataFrame - Training features
    y_train : array - Training target
    
    Returns:
    --------
    ensemble : Trained model
    """
    
    print("\n" + "=" * 80)
    print("STEP 6: TRAINING MODEL")
    print("=" * 80)
    
    print("\n✓ Training ensemble model (this may take 1-2 minutes)...")
    print("  Training on {0} samples with {1} features".format(X_train.shape[0], X_train.shape[1]))
    
    ensemble.fit(X_train, y_train)
    
    print("\n✓ Model training complete!")
    
    return ensemble

# ============================================================================
# STEP 7: 5-FOLD CROSS-VALIDATION
# ============================================================================

def cross_validate_model(ensemble, X_train, y_train):
    """
    Perform 5-fold cross-validation
    
    Parameters:
    -----------
    ensemble : Trained ensemble model
    X_train : DataFrame - Training features
    y_train : array - Training target
    
    Returns:
    --------
    cv_scores : array - Cross-validation scores
    """
    
    print("\n" + "=" * 80)
    print("STEP 7: 5-FOLD CROSS-VALIDATION")
    print("=" * 80)
    
    print("\n✓ Performing 5-fold cross-validation...")
    print("  (Training 5 models, each testing on 1/5 of data)")
    
    cv_scores = cross_val_score(
        ensemble,
        X_train,
        y_train,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    print("\n✓ Cross-validation results:")
    for i, score in enumerate(cv_scores, 1):
        print(f"  - Fold {i}: {score:.4f} ({score*100:.2f}%)")
    
    print(f"\n  - Mean: {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
    print(f"  - Std: {cv_scores.std():.4f} ({cv_scores.std()*100:.2f}%)")
    print(f"  - 95% CI: [{cv_scores.mean()-1.96*cv_scores.std():.4f}, "
          f"{cv_scores.mean()+1.96*cv_scores.std():.4f}]")
    
    print("\n✓ Interpretation: Model is STABLE if std is < 0.01")
    
    return cv_scores

# ============================================================================
# STEP 8: EVALUATE MODEL
# ============================================================================

def evaluate_model(ensemble, X_train, X_test, y_train, y_test):
    """
    Evaluate model on train and test sets
    
    Parameters:
    -----------
    ensemble : Trained model
    X_train, X_test : Feature sets
    y_train, y_test : Target sets
    
    Returns:
    --------
    metrics : dict - Evaluation metrics
    """
    
    print("\n" + "=" * 80)
    print("STEP 8: MODEL EVALUATION")
    print("=" * 80)
    
    # Predictions
    y_train_pred = ensemble.predict(X_train)
    y_test_pred = ensemble.predict(X_test)
    
    # Probabilities for ROC
    y_test_proba = ensemble.predict_proba(X_test)[:, 1]
    
    metrics = {}
    
    # Train metrics
    print("\n📊 TRAINING SET METRICS:")
    print("-" * 80)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"Accuracy:  {train_acc:.4f} ({train_acc*100:.2f}%)")
    
    metrics['train_accuracy'] = train_acc
    
    # Test metrics
    print("\n📊 TEST SET METRICS:")
    print("-" * 80)
    
    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred)
    test_rec = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Precision: {test_prec:.4f} ({test_prec*100:.2f}%)")
    print(f"Recall:    {test_rec:.4f} ({test_rec*100:.2f}%)")
    print(f"F1-Score:  {test_f1:.4f}")
    print(f"ROC-AUC:   {test_auc:.4f}")
    
    metrics['test_accuracy'] = test_acc
    metrics['test_precision'] = test_prec
    metrics['test_recall'] = test_rec
    metrics['test_f1'] = test_f1
    metrics['test_auc'] = test_auc
    
    # Confusion Matrix
    print("\n📊 CONFUSION MATRIX:")
    print("-" * 80)
    
    cm = confusion_matrix(y_test, y_test_pred)
    print("\n           Predicted")
    print("           Negative  Hypothyroid")
    print(f"Actual Negative       {cm[0,0]:5d}      {cm[0,1]:5d}")
    print(f"       Hypothyroid    {cm[1,0]:5d}      {cm[1,1]:5d}")
    
    # Classification Report
    print("\n📊 DETAILED CLASSIFICATION REPORT:")
    print("-" * 80)
    print(classification_report(y_test, y_test_pred, 
                                target_names=['Negative', 'Hypothyroid']))
    
    return metrics

# ============================================================================
# STEP 9: FEATURE IMPORTANCE ANALYSIS
# ============================================================================

def analyze_feature_importance(ensemble, feature_names):
    """
    Extract and display feature importance
    
    Parameters:
    -----------
    ensemble : Trained model
    feature_names : list - Names of features
    
    Returns:
    --------
    importance_df : DataFrame - Ranked features by importance
    """
    
    print("\n" + "=" * 80)
    print("STEP 9: FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)
    
    # Get Random Forest feature importances
    # (it has built-in feature importance)
    rf_model = ensemble.named_estimators_['rf']
    importances = rf_model.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances,
        'Importance_Percent': importances * 100
    }).sort_values('Importance', ascending=False)
    
    print("\n✓ Top 15 Most Important Features:")
    print("-" * 80)
    print(f"{'Rank':<5} {'Feature':<30} {'Importance':<15} {'Percent':<10}")
    print("-" * 80)
    
    for idx, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
        print(f"{idx:<5} {row['Feature']:<30} {row['Importance']:<15.6f} {row['Importance_Percent']:<10.2f}%")
    
    # Summary
    print("\n📊 Feature Importance Summary:")
    print("-" * 80)
    
    # Original vs new features
    original_features = ['TSH', 'T3', 'TT4', 'FTI', 'TSH_TT4_ratio', 'T3_TT4_ratio']
    new_features = ['family_history_level', 'weight_change_kg', 'weight_change_norm']
    
    original_importance = importance_df[importance_df['Feature'].isin(original_features)]['Importance_Percent'].sum()
    new_importance = importance_df[importance_df['Feature'].isin(new_features)]['Importance_Percent'].sum()
    
    print(f"Original features (hormones): {original_importance:.2f}%")
    print(f"New features (genetics+symptoms): {new_importance:.2f}%")
    
    return importance_df

# ============================================================================
# STEP 10: SAVE MODEL & ARTIFACTS
# ============================================================================

def save_model(ensemble, scaler, label_encoder, importance_df):
    """
    Save trained model and preprocessing artifacts
    
    Parameters:
    -----------
    ensemble : Trained model
    scaler : StandardScaler
    label_encoder : LabelEncoder
    importance_df : Feature importance DataFrame
    """
    
    print("\n" + "=" * 80)
    print("STEP 10: SAVING MODEL ARTIFACTS")
    print("=" * 80)
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # Save model
    model_path = "models/thyroid_ensemble_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(ensemble, f)
    print(f"\n✓ Model saved: {model_path}")
    
    # Save scaler
    scaler_path = "models/thyroid_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved: {scaler_path}")
    
    # Save label encoder
    encoder_path = "models/thyroid_label_encoder.pkl"
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"✓ Label encoder saved: {encoder_path}")
    
    # Save feature importance
    importance_path = "outputs/feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"✓ Feature importance saved: {importance_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function
    """
    
    print("\n" + "=" * 80)
    print("THYROID DISEASE PREDICTION: MODEL TRAINING & VALIDATION")
    print("=" * 80)
    print("\nThis script trains a weighted ensemble model with:")
    print("- Random Forest (50%)")
    print("- SVM (30%)")
    print("- KNN (20%)")
    print("=" * 80)
    
    # Step 1: Load data
    df = load_data("outputs/hypothyroid_final_improved.csv")
    
    # Step 2: Prepare data
    X, y, label_encoder, feature_names = prepare_data(df)
    
    # Step 3: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 4: Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Step 5: Build model
    ensemble = build_ensemble_model()
    
    # Step 6: Train model
    ensemble = train_model(ensemble, X_train_scaled, y_train)
    
    # Step 7: Cross-validation
    cv_scores = cross_validate_model(ensemble, X_train_scaled, y_train)
    
    # Step 8: Evaluate
    metrics = evaluate_model(ensemble, X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Step 9: Feature importance
    importance_df = analyze_feature_importance(ensemble, feature_names)
    
    # Step 10: Save model
    save_model(ensemble, scaler, label_encoder, importance_df)
    
    # Final summary
    print("\n" + "=" * 80)
    print("MODEL TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\n✓ Test Accuracy: {metrics['test_accuracy']:.4f} ({metrics['test_accuracy']*100:.2f}%)")
    print(f"✓ Test Precision: {metrics['test_precision']:.4f}")
    print(f"✓ Test Recall: {metrics['test_recall']:.4f}")
    print(f"✓ Test F1-Score: {metrics['test_f1']:.4f}")
    print(f"\n✓ Cross-validation: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"\n✓ Model saved to: models/thyroid_ensemble_model.pkl")
    print(f"✓ Scaler saved to: models/thyroid_scaler.pkl")
    print(f"✓ Ready for deployment!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
