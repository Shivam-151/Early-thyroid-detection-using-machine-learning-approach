"""Corrected thyroid research pipeline.

IMPORTANT: family_history_level and weight_change_kg are SIMULATED features.
They are generated independently of diagnosis and must not be interpreted as
observed clinical measurements. The default weight-change distribution below
is an explicit simulation assumption, not an empirical clinical estimate.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)
from sklearn.inspection import permutation_importance

SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
FAMILY_HISTORY_PROBS = [0.60, 0.20, 0.10, 0.10]  # levels 0,1,2,3; same for every row
WEIGHT_CHANGE_MEAN_KG = 0.5  # simulation assumption; confirm/replace before publication
WEIGHT_CHANGE_SD_KG = 1.0    # simulation assumption; confirm/replace before publication


def resolve_paths():
    here = Path(__file__).resolve().parent
    candidates = [
        here / "outputs" / "hypothyroid_cleaned_correct.csv",
        here / "hypothyroid_cleaned_correct.csv",
        here.parent / "outputs" / "hypothyroid_cleaned_correct.csv",
    ]
    for path in candidates:
        if path.exists():
            return path, path.parent
    raise FileNotFoundError("Could not find outputs/hypothyroid_cleaned_correct.csv. Put this script in the project root or update INPUT_FILE.")


def make_features(df):
    df = df.copy()
    if 'class' not in df.columns:
        raise ValueError("Input CSV must contain a 'class' target column.")
    df['class'] = df['class'].astype(str).str.strip().str.lower()
    allowed = {'hypothyroid', 'negative'}
    if not set(df['class'].dropna().unique()).issubset(allowed):
        raise ValueError(f"Unexpected class labels: {df['class'].unique().tolist()}")

    rng = np.random.default_rng(SEED)
    # One common probability distribution for every row; diagnosis is not passed in.
    df['family_history_level'] = rng.choice([0, 1, 2, 3], size=len(df), p=FAMILY_HISTORY_PROBS)
    # One common distribution for every row; no diagnosis- or age-conditioned sampling.
    df['weight_change_kg'] = rng.normal(WEIGHT_CHANGE_MEAN_KG, WEIGHT_CHANGE_SD_KG, size=len(df))

    # Ratios are engineered from the original measured variables. Imputation is done
    # inside each model pipeline; invalid/zero-denominator ratios are set to missing.
    if {'TSH', 'TT4'}.issubset(df.columns):
        denom = pd.to_numeric(df['TT4'], errors='coerce').replace(0, np.nan)
        df['TSH_TT4_ratio'] = pd.to_numeric(df['TSH'], errors='coerce') / denom
    if {'T3', 'TT4'}.issubset(df.columns):
        denom = pd.to_numeric(df['TT4'], errors='coerce').replace(0, np.nan)
        df['T3_TT4_ratio'] = pd.to_numeric(df['T3'], errors='coerce') / denom
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def build_model(seed=SEED):
    # This dataset is numeric after the cleaning step. Imputation and scaling are
    # fitted separately within each CV fold / training split to avoid preprocessing leakage.
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    rf = Pipeline([('prep', clone(preprocessor)), ('model', RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_split=5, random_state=seed,
        class_weight=None, n_jobs=-1))])
    svm = Pipeline([('prep', clone(preprocessor)), ('model', SVC(
        kernel='rbf', gamma='scale', probability=True, random_state=seed))])
    knn = Pipeline([('prep', clone(preprocessor)), ('model', KNeighborsClassifier(
        n_neighbors=5, weights='distance', n_jobs=-1))])
    return VotingClassifier(
        estimators=[('rf', rf), ('svm', svm), ('knn', knn)],
        voting='soft', weights=[0.5, 0.3, 0.2], n_jobs=None)


def main():
    input_path, output_dir = resolve_paths()
    output_dir.mkdir(parents=True, exist_ok=True)
    outdir = output_dir / 'corrected_research_run'
    outdir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_path)
    if 'class' not in raw.columns:
        raise ValueError("No 'class' column in input data.")
    print(f"Input: {input_path}\nRows: {len(raw)}\nOriginal columns: {len(raw.columns)}")
    print('Original class counts:', raw['class'].value_counts(dropna=False).to_dict())

    df = make_features(raw)
    # Target convention: hypothyroid=1 (positive class), negative=0.
    y = (df['class'] == 'hypothyroid').astype(int)
    X = df.drop(columns=['class'])
    # Ensure every predictor is numeric for this cleaned dataset.
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.replace([np.inf, -np.inf], np.nan)

    # Explicit audit: synthetic feature distribution should not be label-conditioned.
    audit = df.groupby('class').agg(
        n=('family_history_level', 'size'),
        family_history_mean=('family_history_level', 'mean'),
        weight_change_mean_kg=('weight_change_kg', 'mean'),
        weight_change_sd_kg=('weight_change_kg', 'std'))
    print('\nSIMULATION AUDIT (descriptive only; not clinical evidence):')
    print(audit.to_string())
    print('\nFamily-history generation probabilities used for ALL rows:', FAMILY_HISTORY_PROBS)
    print(f'Weight-change distribution used for ALL rows: Normal({WEIGHT_CHANGE_MEAN_KG}, {WEIGHT_CHANGE_SD_KG}^2) kg')

    # These tests are diagnostic descriptions only. Since features are simulated
    # independently of diagnosis, do not interpret p-values as clinical evidence.
    contingency = pd.crosstab(df['family_history_level'], df['class'])
    chi2, chi_p, chi_dof, _ = chi2_contingency(contingency)
    rho, rho_p = spearmanr(df['weight_change_kg'], y)
    print(f'Family-history chi-square (simulation diagnostic): chi2={chi2:.4f}, df={chi_dof}, p={chi_p:.4g}')
    print(f'Weight-change Spearman (simulation diagnostic): rho={rho:.4f}, p={rho_p:.4g}')
    print('Interpretation: these tests do NOT establish real-world clinical associations.')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    model = build_model()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, list(model.classes_).index(1)]

    cm = confusion_matrix(y_test, pred, labels=[1, 0])  # positive hypothyroid first
    metrics = {
        'n_rows': int(len(df)),
        'n_hypothyroid': int(y.sum()),
        'n_negative': int((y == 0).sum()),
        'train_rows': int(len(X_train)),
        'test_rows': int(len(X_test)),
        'positive_class': 'hypothyroid',
        'accuracy': float(accuracy_score(y_test, pred)),
        'precision_hypothyroid': float(precision_score(y_test, pred, pos_label=1, zero_division=0)),
        'recall_hypothyroid': float(recall_score(y_test, pred, pos_label=1, zero_division=0)),
        'f1_hypothyroid': float(f1_score(y_test, pred, pos_label=1, zero_division=0)),
        'roc_auc_hypothyroid': float(roc_auc_score(y_test, proba)),
        'confusion_matrix_order_hypothyroid_then_negative': cm.tolist(),
        'family_history_chi2_diagnostic': float(chi2),
        'family_history_chi2_p_diagnostic': float(chi_p),
        'weight_change_spearman_diagnostic': float(rho),
        'weight_change_spearman_p_diagnostic': float(rho_p),
        'seed': SEED,
        'family_history_probs_common_all_rows': FAMILY_HISTORY_PROBS,
        'weight_change_distribution_common_all_rows': {'distribution': 'normal', 'mean_kg': WEIGHT_CHANGE_MEAN_KG, 'sd_kg': WEIGHT_CHANGE_SD_KG},
        'warning': 'Simulated features are not observed clinical measurements; results are not clinical validation.'
    }
    print('\nHELD-OUT TEST RESULTS (hypothyroid is positive class):')
    for k in ['accuracy','precision_hypothyroid','recall_hypothyroid','f1_hypothyroid','roc_auc_hypothyroid']:
        print(f'{k}: {metrics[k]:.4f}')
    print('Confusion matrix [rows=true, columns=predicted], order [hypothyroid, negative]:')
    print(cm)
    print('\nClassification report (hypothyroid=1):')
    print(classification_report(y_test, pred, labels=[1,0], target_names=['hypothyroid','negative'], zero_division=0))

    # Stratified CV: the full imputation/scaling/model pipeline is refit per fold.
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    cv_result = cross_validate(build_model(), X_train, y_train, cv=cv,
        scoring={'accuracy':'accuracy', 'precision':'precision', 'recall':'recall', 'f1':'f1', 'roc_auc':'roc_auc'},
        n_jobs=1, error_score='raise')
    cv_summary = {f'{metric}_fold_scores': values.tolist() for metric, values in cv_result.items() if metric.startswith('test_')}
    for metric, values in cv_result.items():
        if metric.startswith('test_'):
            name = metric.replace('test_', '')
            cv_summary[f'{name}_mean'] = float(np.mean(values))
            cv_summary[f'{name}_sd'] = float(np.std(values, ddof=1))
    metrics['cross_validation'] = cv_summary
    print('\n5-FOLD STRATIFIED CV (training split only):')
    for name in ['accuracy','precision','recall','f1','roc_auc']:
        print(f'{name}: {cv_summary[name + "_mean"]:.4f} ± {cv_summary[name + "_sd"]:.4f}')

    # Fit a separate RF pipeline for permutation importance on held-out data.
    # Permutation importance is used because the model is a voting ensemble.
    importance = permutation_importance(model, X_test, y_test, n_repeats=5,
        random_state=SEED, scoring='roc_auc', n_jobs=1)
    imp_df = pd.DataFrame({'feature': X.columns,
        'permutation_importance_mean': importance.importances_mean,
        'permutation_importance_sd': importance.importances_std}).sort_values('permutation_importance_mean', ascending=False)
    imp_df.to_csv(outdir / 'permutation_importance.csv', index=False)
    df.to_csv(outdir / 'thyroid_enhanced_simulated.csv', index=False)
    with open(outdir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    with open(outdir / 'model.pkl', 'wb') as f:
        import pickle; pickle.dump(model, f)
    print(f'\nSaved outputs to: {outdir}')
    print('NOTE: do not copy previous manuscript metrics into the paper; use this run’s outputs after methodological review.')

if __name__ == '__main__':
    main()
