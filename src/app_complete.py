"""
STREAMLIT APP: THYROID DISEASE PREDICTION INTERFACE
===================================================

Interactive web application for thyroid disease prediction
Uses trained ensemble model with family history and weight change features

Run with: streamlit run app_complete.py

Author: Medical ML Team
Date: 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Thyroid Disease Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR BETTER UI
# ============================================================================

def load_custom_css():
    """Apply custom styling"""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .danger-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS AND PREPROCESSING ARTIFACTS
# ============================================================================

@st.cache_resource
def load_models():
    """Load trained model and preprocessing artifacts"""
    
    models_dir = Path('models')
    
    try:
        # Load model
        model_path = models_dir / 'thyroid_ensemble_model.pkl'
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler
        scaler_path = models_dir / 'thyroid_scaler.pkl'
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load label encoder
        encoder_path = models_dir / 'thyroid_label_encoder.pkl'
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Load feature names (if saved during training)
        feature_names_path = models_dir / 'feature_names.pkl'
        if feature_names_path.exists():
            with open(feature_names_path, 'rb') as f:
                feature_names = pickle.load(f)
        else:
            # Default feature names (should match training)
            feature_names = ['age', 'TSH', 'T3', 'TT4', 'FTI', 
                           'family_history_level', 'weight_change_kg', 
                           'weight_change_norm', 'TSH_TT4_ratio', 'T3_TT4_ratio']
        
        return model, scaler, label_encoder, feature_names
    
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()

# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================

def calculate_derived_features(tsh, t3, tt4, fti, weight_change):
    """
    Calculate derived features matching training pipeline
    
    Args:
        tsh: TSH level
        t3: T3 level
        tt4: TT4 level
        fti: FTI level
        weight_change: Weight change in kg
    
    Returns:
        dict: Derived features
    """
    
    # Avoid division by zero
    epsilon = 1e-6
    
    features = {
        'TSH_TT4_ratio': tsh / (tt4 + epsilon),
        'T3_TT4_ratio': t3 / (tt4 + epsilon),
        'weight_change_norm': (weight_change + 10) / 20  # Normalize to [0, 1] range
    }
    
    return features

def prepare_input_features(age, sex, tsh, t3, tt4, fti, fh_level, weight_change, 
                          on_thyroxine, thyroid_surgery, goitre, feature_names):
    """
    Prepare input features for prediction
    
    Args:
        Various patient parameters
        feature_names: List of feature names from training
    
    Returns:
        pd.DataFrame: Prepared features
    """
    
    # Calculate derived features
    derived = calculate_derived_features(tsh, t3, tt4, fti, weight_change)
    
    # Base features
    features = {
        'age': age,
        'sex': 1 if sex == 'Male' else 0,
        'TSH': tsh,
        'T3': t3,
        'TT4': tt4,
        'FTI': fti,
        'family_history_level': fh_level,
        'weight_change_kg': weight_change,
        'weight_change_norm': derived['weight_change_norm'],
        'TSH_TT4_ratio': derived['TSH_TT4_ratio'],
        'T3_TT4_ratio': derived['T3_TT4_ratio'],
        'on_thyroxine': 1 if on_thyroxine else 0,
        'thyroid_surgery': 1 if thyroid_surgery else 0,
        'goitre': 1 if goitre else 0,
    }
    
    # Create DataFrame with all required features
    df = pd.DataFrame([features])
    
    # Ensure all feature_names are present
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    # Return in correct order
    return df[feature_names]

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_inputs(tsh, t3, tt4, fti, weight_change):
    """
    Validate user inputs for clinical plausibility
    
    Returns:
        tuple: (is_valid, warnings_list)
    """
    
    warnings = []
    
    # TSH validation
    if tsh < 0.1 or tsh > 50:
        warnings.append("⚠️ TSH level is outside typical range (0.1-50 μIU/mL)")
    
    # T3 validation
    if t3 < 0.5 or t3 > 5:
        warnings.append("⚠️ T3 level is outside typical range (0.5-5 ng/dL)")
    
    # TT4 validation
    if tt4 < 20 or tt4 > 200:
        warnings.append("⚠️ TT4 level is outside typical range (20-200 ng/dL)")
    
    # FTI validation
    if fti < 20 or fti > 200:
        warnings.append("⚠️ FTI level is outside typical range (20-200 ng/dL)")
    
    # Weight change validation
    if abs(weight_change) > 15:
        warnings.append("⚠️ Extreme weight change detected (>15 kg)")
    
    return len(warnings) == 0, warnings

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_probability_gauge(prob_hypothyroid):
    """Create a visual gauge for probability"""
    
    import plotly.graph_objects as go
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob_hypothyroid * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Hypothyroid Risk (%)", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d4edda'},
                {'range': [30, 70], 'color': '#fff3cd'},
                {'range': [70, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_feature_importance_chart(feature_values):
    """Create horizontal bar chart of feature values"""
    
    import plotly.graph_objects as go
    
    fig = go.Figure(go.Bar(
        x=list(feature_values.values()),
        y=list(feature_values.keys()),
        orientation='h',
        marker=dict(
            color=list(feature_values.values()),
            colorscale='Blues',
            showscale=True
        )
    ))
    
    fig.update_layout(
        title="Feature Values (Standardized)",
        xaxis_title="Value",
        yaxis_title="Feature",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application"""
    
    # Load custom CSS
    load_custom_css()
    
    # Title
    st.markdown("""
    <div class="main-header">
        🏥 Thyroid Disease Prediction System
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### AI-Powered Clinical Decision Support
    
    This application predicts thyroid disease risk using an **ensemble machine learning model** trained on:
    - 🔬 **Hormone measurements** (TSH, T3, TT4, FTI)
    - 🧬 **Genetic information** (Family history)
    - 📊 **Clinical symptoms** (Weight change, medical history)
    - 🤖 **Weighted ensemble** (Random Forest + SVM + KNN)
    
    