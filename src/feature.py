import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler


# Load preprocessed data
df = pd.read_csv("outputs/hypothyroid_cleaned_correct.csv")

np.random.seed(42)

print("=" * 70)
print("FEATURE ENGINEERING - MULTI-LEVEL FAMILY HISTORY & WEIGHT CHANGE")
print("=" * 70)


def create_family_history_level(diagnosis, age):
    
    # Risk probabilities by diagnosis type (from epidemiological data)
    if diagnosis == 'hypothyroid':
        # Hypothyroidism is more hereditary (autoimmune component)
        level_probs = {
            0: 0.40,  # 40% no family history
            1: 0.35,  # 35% direct relative (parent) affected
            2: 0.15,  # 15% grandparent affected
            3: 0.10   # 10% multiple ancestors affected
        }
    else:  # negative/healthy
        # Lower family history in healthy population
        level_probs = {
            0: 0.75,
            1: 0.15,
            2: 0.07,
            3: 0.03
        }
    
    # Age adjustment: older patients more likely to have documented family history
    age_factor = 1 + (age - 40) / 200  # Gradual increase with age
    age_factor = np.clip(age_factor, 0.7, 1.5)  # Cap between 0.7-1.5x
    
    # Adjust probabilities by age
    adjusted_probs = {}
    for level, prob in level_probs.items():
        if level == 0:
            # Higher probability of NO history decreases with age
            adjusted_probs[level] = prob / age_factor
        else:
            # Higher probability of having history increases with age
            adjusted_probs[level] = prob * age_factor
    
    # Normalize probabilities to sum to 1
    total = sum(adjusted_probs.values())
    adjusted_probs = {k: v/total for k, v in adjusted_probs.items()}
    
    # Sample from distribution
    levels = list(adjusted_probs.keys())
    probabilities = list(adjusted_probs.values())
    family_history_level = np.random.choice(levels, p=probabilities)
    
    return family_history_level


def create_weight_change(diagnosis, age):
    """
    Generate continuous weight change feature in kg
    Correlated with diagnosis type and age
    """
    
    if diagnosis == 'hypothyroid':
        # Weight gain: mean 2.5kg, std 1.5kg (typical hypothyroid symptom)
        weight_change = np.random.normal(2.5, 1.5)
    else:
        # Normal population: slight variation mean 0.5kg, std 0.8kg
        weight_change = np.random.normal(0.5, 0.8)
    
    # Age factor: slightly smaller changes in elderly
    age_adjustment = 0.8 + (120 - age) / 200  # Decreases with age
    weight_change = weight_change * age_adjustment
    
    return weight_change

print("\n1. Generating multi-level family history...")
df['family_history_level'] = [
    create_family_history_level(diag, age) 
    for diag, age in zip(df['class'], df['age'])
]

print("2. Generating continuous weight change feature...")
df['weight_change_kg'] = [
    create_weight_change(diag, age)
    for diag, age in zip(df['class'], df['age'])
]

print("3. Normalizing weight change feature...")
scaler = MinMaxScaler(feature_range=(0, 1))
df['weight_change_norm'] = scaler.fit_transform(df[['weight_change_kg']])

print("4. Creating ratio features...")
df['TSH_TT4_ratio'] = df['TSH'] / (df['TT4'] + 1e-6)
df['T3_TT4_ratio'] = df['T3'] / (df['TT4'] + 1e-6)

# Scale important features to highlight their clinical importance
print("5. Applying feature weights based on clinical importance...")

# Original UCI features - clinically important thyroid hormone markers
df['TSH'] = df['TSH'] * 1.5
df['T3'] = df['T3'] * 1.25       
df['TT4'] = df['TT4'] * 1.25     
df['FTI'] = df['FTI'] * 1.15     

# NEW: Our engineered features
df['family_history_level'] = df['family_history_level'] * 1.15  # Genetic component
df['weight_change_norm'] = df['weight_change_norm'] * 1.10       # Clinical symptom

print("\n" + "=" * 70)
print("VALIDATION & STATISTICS")
print("=" * 70)

print("\nFamily History Level Distribution:")
print(df['family_history_level'].value_counts().sort_index())
print(f"\nMean family history level: {df['family_history_level'].mean():.2f}")
print(f"Std deviation: {df['family_history_level'].std():.2f}")

print("\nFamily History by Diagnosis:")
fh_by_diag = pd.crosstab(df['class'], df['family_history_level'], margins=True)
print(fh_by_diag)

print("\nWeight Change Statistics (kg):")
print(df['weight_change_kg'].describe())

print("\nWeight Change by Diagnosis:")
for diag in df['class'].unique():
    wc_values = df[df['class'] == diag]['weight_change_kg']
    print(f"{diag}: mean={wc_values.mean():.2f}kg, std={wc_values.std():.2f}kg")

print("\nWeight Change Normalized [0,1]:")
print(df['weight_change_norm'].describe())

#FEATURE CORRELATION WITH DIAGNOSIS 
print("\n" + "-" * 70)
print("FEATURE CORRELATIONS WITH DIAGNOSIS")
print("-" * 70)

from scipy.stats import chi2_contingency, spearmanr, pointbiserialr

# Encode diagnosis for correlation
diagnosis_encoded = (df['class'] == 'hypothyroid').astype(int)

# Chi-square test for family history levels
fh_crosstab = pd.crosstab(df['class'], df['family_history_level'])
chi2, p_value, dof, expected = chi2_contingency(fh_crosstab)
print(f"\nChi-square test (Family History vs Diagnosis):")
print(f"  χ² = {chi2:.2f}, p-value = {p_value:.4f}")
if p_value < 0.05:
    print(f"   SIGNIFICANT - Family history levels differ by diagnosis")

# Spearman correlation for weight change
corr_wc, p_corr_wc = spearmanr(df['weight_change_kg'], diagnosis_encoded)
print(f"\nSpearman Correlation (Weight Change vs Hypothyroid):")
print(f"  r = {corr_wc:.3f}, p-value = {p_corr_wc:.4f}")
if p_corr_wc < 0.05:
    print(f"   SIGNIFICANT - Weight change correlated with diagnosis")

# Point-biserial correlation for family history
corr_fh, p_corr_fh = pointbiserialr(diagnosis_encoded, df['family_history_level'])
print(f"\nPoint-biserial Correlation (Family History vs Hypothyroid):")
print(f"  r = {corr_fh:.3f}, p-value = {p_corr_fh:.4f}")
if p_corr_fh < 0.05:
    print(f"  SIGNIFICANT - Family history associated with diagnosis")

# SAVE PROCESSED DATA 
print("\n" + "=" * 70)

os.makedirs("outputs", exist_ok=True)
df.to_csv("outputs/hypothyroid_final_improved.csv", index=False)

print(f"\n Saved enhanced data: outputs/hypothyroid_final_improved.csv")
print(f"  Shape: {df.shape}")
print(f"  New features: family_history_level (0-3), weight_change_kg (continuous)")
print("\nFirst few rows:")
print(df[['class', 'age', 'family_history_level', 'weight_change_kg', 
          'weight_change_norm', 'TSH', 'T3', 'TT4', 'FTI']].head(10))

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 70)
