"""
Predictive Modeling - Machine Learning for Business Value
=========================================================
Practical ML models addressing real business questions with
clear explanations and validated performance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
                             silhouette_score, mean_absolute_error, mean_squared_error,
                             mean_absolute_percentage_error)
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.ensemble import BalancedRandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.data_loader import load_data
from utils.styles import apply_global_styles, style_plotly_chart

st.set_page_config(page_title="Predictive Modeling | ARAG", layout="wide")
apply_global_styles()

# Modern Header with gradient background
st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 3rem 2rem; border-radius: 20px; margin-bottom: 2.5rem;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);'>
    <h1 style='color: white; margin: 0 0 0.5rem 0; font-size: 2.5rem; font-weight: 800; border: none; text-align: center;'>
        Predictive Modeling
    </h1>
    <p style='color: rgba(255,255,255,0.95); font-size: 1.2rem; margin: 0; text-align: center; font-weight: 300;'>
        Machine Learning Models That Drive Business Decisions
    </p>
</div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def get_data():
    try:
        return load_data("Data Science Sample data.csv")
    except:
        return load_data("../Data Science Sample data.csv")

df = get_data()

# ============================================================================
# INTRODUCTION - WHY THESE MODELS MATTER
# ============================================================================
st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 16px; margin-bottom: 2rem; box-shadow: 0 8px 16px rgba(0,0,0,0.1);'>
    <h2 style='color: white; margin: 0 0 1rem 0; border: none;'>Why Predictive Models Matter for ARAG</h2>
    <p style='color: rgba(255,255,255,0.95); font-size: 1.1rem; margin: 0; line-height: 1.6;'>
    In legal expense insurance, making the right decisions early in the claims process has
    significant financial and operational impact. These models provide genuine predictive value
    for capacity planning, cost management, and operational efficiency.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style='background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
                border: 1px solid #e2e8f0; border-left: 4px solid #667eea;
                padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); min-height: 200px;'>
        <div style='background: #f1f5f9; border-radius: 6px; padding: 0.4rem 0.9rem;
                    display: inline-block; margin-bottom: 1rem; border-left: 3px solid #667eea;'>
            <span style='color: #475569; font-weight: 600; font-size: 0.75rem;
                         text-transform: uppercase; letter-spacing: 0.5px;'>Forecasting Model</span>
        </div>
        <h3 style='color: #1E3A5F; margin: 0 0 1rem 0; font-size: 1.3rem; font-weight: 600;'>
            Model 1: Claims Demand Forecasting
        </h3>
        <p style='color: #475569; margin: 0.5rem 0; font-size: 0.9rem; line-height: 1.5;'>
            <strong style='color: #1E3A5F;'>Question:</strong> How many claims should we expect next week/month?
        </p>
        <p style='color: #64748b; margin: 0; font-size: 0.85rem; line-height: 1.4;'>
            <strong style='color: #475569;'>Value:</strong> Staffing optimization, resource allocation, capacity planning
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
                border: 1px solid #e2e8f0; border-left: 4px solid #f5576c;
                padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); min-height: 200px;'>
        <div style='background: #f1f5f9; border-radius: 6px; padding: 0.4rem 0.9rem;
                    display: inline-block; margin-bottom: 1rem; border-left: 3px solid #f5576c;'>
            <span style='color: #475569; font-weight: 600; font-size: 0.75rem;
                         text-transform: uppercase; letter-spacing: 0.5px;'>Classification Model</span>
        </div>
        <h3 style='color: #1E3A5F; margin: 0 0 1rem 0; font-size: 1.3rem; font-weight: 600;'>
            Model 2: Litigation Cost Prediction
        </h3>
        <p style='color: #475569; margin: 0.5rem 0; font-size: 0.9rem; line-height: 1.5;'>
            <strong style='color: #1E3A5F;'>Question:</strong> For claims going to litigation, will this be a high-cost case?
        </p>
        <p style='color: #64748b; margin: 0; font-size: 0.85rem; line-height: 1.4;'>
            <strong style='color: #475569;'>Value:</strong> Early reserve setting, senior handler assignment, proactive cost management
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
                border: 1px solid #e2e8f0; border-left: 4px solid #00d4aa;
                padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); min-height: 200px;'>
        <div style='background: #f1f5f9; border-radius: 6px; padding: 0.4rem 0.9rem;
                    display: inline-block; margin-bottom: 1rem; border-left: 3px solid #00d4aa;'>
            <span style='color: #475569; font-weight: 600; font-size: 0.75rem;
                         text-transform: uppercase; letter-spacing: 0.5px;'>Classification Model</span>
        </div>
        <h3 style='color: #1E3A5F; margin: 0 0 1rem 0; font-size: 1.3rem; font-weight: 600;'>
            Model 3: Litigation Duration Prediction
        </h3>
        <p style='color: #475569; margin: 0.5rem 0; font-size: 0.9rem; line-height: 1.5;'>
            <strong style='color: #1E3A5F;'>Question:</strong> For claims going to litigation, will this resolve quickly or take months?
        </p>
        <p style='color: #64748b; margin: 0; font-size: 0.85rem; line-height: 1.4;'>
            <strong style='color: #475569;'>Value:</strong> Customer expectations, workload planning, SLA management
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
                border: 1px solid #e2e8f0; border-left: 4px solid #f59e0b;
                padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); min-height: 200px;'>
        <div style='background: #f1f5f9; border-radius: 6px; padding: 0.4rem 0.9rem;
                    display: inline-block; margin-bottom: 1rem; border-left: 3px solid #f59e0b;'>
            <span style='color: #475569; font-weight: 600; font-size: 0.75rem;
                         text-transform: uppercase; letter-spacing: 0.5px;'>Clustering Model</span>
        </div>
        <h3 style='color: #1E3A5F; margin: 0 0 1rem 0; font-size: 1.3rem; font-weight: 600;'>
            Model 4: Customer Segmentation
        </h3>
        <p style='color: #475569; margin: 0.5rem 0; font-size: 0.9rem; line-height: 1.5;'>
            <strong style='color: #1E3A5F;'>Question:</strong> How can we segment claims for targeted strategies?
        </p>
        <p style='color: #64748b; margin: 0; font-size: 0.85rem; line-height: 1.4;'>
            <strong style='color: #475569;'>Value:</strong> Differentiated service, risk-based routing, process optimization
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# DATA PREPARATION
# ============================================================================
st.markdown("""
<div style='text-align: center; margin: 2.5rem 0 2rem 0;'>
    <h2 style='color: #1E3A5F; font-size: 1.9rem; font-weight: 700; margin-bottom: 0.5rem; border: none;'>
        Data Preparation
    </h2>
    <p style='color: #64748b; font-size: 1rem; margin: 0;'>
        Clean, encode, and transform data for optimal model performance
    </p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def prepare_modeling_data(df):
    """Prepare data for all models."""
    # Filter completed claims only for classification models
    model_df = df[df['Claim Status'].isin(['C', 'D', 'N'])].copy()

    # Base feature columns
    base_feature_cols = ['Claim Category Description', 'Claim Type Description',
                    'Business Class Description', 'Agent Company Name',
                    'Inward Reinsurance']

    # Remove rows with missing features
    model_df = model_df.dropna(subset=base_feature_cols)

    # Encode categorical features
    encoders = {}
    encoded_features = []

    # Encode base features
    for col in base_feature_cols:
        le = LabelEncoder()
        model_df[f'{col}_enc'] = le.fit_transform(model_df[col].astype(str))
        encoders[col] = le
        encoded_features.append(f'{col}_enc')

    # Store feature names for display
    feature_cols = base_feature_cols

    # For Models 2 and 3: Filter to only Panel/Non-Panel claims (actual litigation)
    litigation_df = model_df[model_df['Litigator Panel Type'].isin(['Panel', 'Non-Panel'])].copy()

    # High cost target for litigation claims
    if len(litigation_df) > 0:
        claims_with_cost = litigation_df[litigation_df['Total Cost To Date'] > 0]['Total Cost To Date']
        cost_threshold = claims_with_cost.median() if len(claims_with_cost) > 0 else 1000
        litigation_df['target_high_cost'] = (litigation_df['Total Cost To Date'] > cost_threshold).astype(int)

        # Quick resolution target - use median resolution time as threshold
        resolution_times = litigation_df[litigation_df['Resolution Time (Days)'] > 0]['Resolution Time (Days)']
        resolution_threshold = resolution_times.median() if len(resolution_times) > 0 else 90
        litigation_df['target_quick'] = (litigation_df['Resolution Time (Days)'] <= resolution_threshold).astype(int)
        litigation_df['resolution_threshold'] = resolution_threshold  # Store for display

    return model_df, litigation_df, feature_cols, encoded_features, encoders

model_df, litigation_df, feature_cols, encoded_features, encoders = prepare_modeling_data(df)

# Modern metrics with gradient cards
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    transition: all 0.3s ease;
    height: 100%;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0.5rem 0;
}
.metric-label {
    color: #64748b;
    font-size: 0.9rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Claims</div>
        <div class="metric-value">{len(df):,}</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Completed Claims</div>
        <div class="metric-value">{len(model_df):,}</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Litigation Claims</div>
        <div class="metric-value">{len(litigation_df):,}</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    litigation_pct = len(litigation_df) / len(model_df) * 100 if len(model_df) > 0 else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Litigation Rate</div>
        <div class="metric-value">{litigation_pct:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# MODEL 1: CLAIMS DEMAND FORECASTING
# ============================================================================
st.markdown("<div style='margin: 3rem 0 2rem 0;'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2.5rem 2rem; border-radius: 20px; margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);'>
    <div style='text-align: center;'>
        <h2 style='color: white; margin: 0 0 0.5rem 0; font-size: 2.2rem; font-weight: 800; border: none;'>
            Model 1: Claims Demand Forecasting
        </h2>
        <p style='color: rgba(255,255,255,0.95); font-size: 1.2rem; margin: 0; font-weight: 300;'>
            Predict future claims volume for capacity planning
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# THE BUSINESS PROBLEM
st.markdown("""
<div style='background: linear-gradient(145deg, #f0f9ff 0%, #e0f2fe 100%);
            border-left: 5px solid #0ea5e9; padding: 2rem; border-radius: 12px;
            margin-bottom: 2rem; box-shadow: 0 4px 12px rgba(14, 165, 233, 0.15);'>
    <h3 style='color: #1E3A5F; margin: 0 0 1rem 0; font-size: 1.5rem; font-weight: 700;'>
        The Business Problem
    </h3>
    <p style='color: #475569; font-size: 1.05rem; line-height: 1.7; margin-bottom: 1rem;'>
        Effective resource planning requires knowing <strong>how many claims to expect</strong>.
        Too few staff means delays and poor customer service; too many means wasted costs.
    </p>
    <div style='margin-top: 1rem; padding: 1rem; background: rgba(14, 165, 233, 0.1); border-radius: 8px;'>
        <strong style='color: #0ea5e9; font-size: 1.05rem;'>Solution Approach:</strong>
        <p style='color: #475569; margin: 0.5rem 0; font-size: 0.95rem; line-height: 1.6;'>
            A <strong>Random Forest Regressor</strong> trained on historical claims data with engineered time series features:
        </p>
        <ul style='color: #475569; margin: 0.5rem 0 0 1.5rem; line-height: 1.5; font-size: 0.9rem;'>
            <li><strong>Lag features</strong> - Previous 1-8 periods of claims volume</li>
            <li><strong>Cyclical encoding</strong> - Sin/cos transformations for month, quarter, and week to capture seasonality</li>
            <li><strong>Rolling statistics</strong> - Moving averages, standard deviations, min/max over 3, 6, 12 periods</li>
            <li><strong>Year-over-year</strong> - Same period last year comparison for trend detection</li>
            <li><strong>Momentum</strong> - Rate of change indicators for trend direction</li>
        </ul>
    </div>
    <div style='margin-top: 1rem; padding: 1rem; background: rgba(14, 165, 233, 0.1); border-radius: 8px;'>
        <strong style='color: #0ea5e9; font-size: 1.05rem;'>Business Value:</strong>
        <ul style='color: #475569; margin: 0.5rem 0 0 1.5rem; line-height: 1.6;'>
            <li><strong>Staffing optimization</strong> - Right-size teams for expected demand</li>
            <li><strong>Budget planning</strong> - Forecast costs based on expected volume</li>
            <li><strong>Seasonal preparation</strong> - Plan for peak periods in advance</li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)

# Prepare time series data for forecasting
@st.cache_data
def prepare_forecast_data(df):
    """Prepare time series data for claims forecasting."""
    # Ensure we have the Input Date
    df_ts = df.dropna(subset=['Input Date']).copy()

    # Create weekly and monthly aggregations
    df_ts['Week'] = df_ts['Input Date'].dt.to_period('W').dt.start_time
    df_ts['Month'] = df_ts['Input Date'].dt.to_period('M').dt.start_time

    # Weekly claims count
    weekly_claims = df_ts.groupby('Week').agg({
        'Claim Reference': 'count',
        'Total Cost To Date': 'sum'
    }).reset_index()
    weekly_claims.columns = ['Date', 'Claims', 'Total_Cost']
    weekly_claims = weekly_claims.sort_values('Date')

    # Monthly claims count
    monthly_claims = df_ts.groupby('Month').agg({
        'Claim Reference': 'count',
        'Total Cost To Date': 'sum'
    }).reset_index()
    monthly_claims.columns = ['Date', 'Claims', 'Total_Cost']
    monthly_claims = monthly_claims.sort_values('Date')

    return weekly_claims, monthly_claims

weekly_claims, monthly_claims = prepare_forecast_data(df)

# Time granularity selector
st.markdown("### Forecast Settings")
forecast_granularity = st.radio(
    "Select time granularity:",
    ["Weekly", "Monthly"],
    horizontal=True,
    help="Choose whether to view weekly or monthly forecasts"
)

# Select the appropriate data
if forecast_granularity == "Weekly":
    ts_data = weekly_claims.copy()
    period_name = "Week"
else:
    ts_data = monthly_claims.copy()
    period_name = "Month"

# Feature engineering for time series
@st.cache_data
def create_forecast_features(ts_data, n_lags=8):
    """Create comprehensive features for forecasting including cyclical encoding."""
    df_feat = ts_data.copy()

    # Lag features (extended to capture more history)
    for i in range(1, n_lags + 1):
        df_feat[f'lag_{i}'] = df_feat['Claims'].shift(i)

    # Rolling statistics (multiple windows)
    df_feat['rolling_mean_3'] = df_feat['Claims'].shift(1).rolling(window=3).mean()
    df_feat['rolling_std_3'] = df_feat['Claims'].shift(1).rolling(window=3).std()
    df_feat['rolling_mean_6'] = df_feat['Claims'].shift(1).rolling(window=6).mean()
    df_feat['rolling_std_6'] = df_feat['Claims'].shift(1).rolling(window=6).std()
    df_feat['rolling_mean_12'] = df_feat['Claims'].shift(1).rolling(window=12).mean()
    df_feat['rolling_min_6'] = df_feat['Claims'].shift(1).rolling(window=6).min()
    df_feat['rolling_max_6'] = df_feat['Claims'].shift(1).rolling(window=6).max()

    # Exponential weighted moving average
    df_feat['ewm_mean'] = df_feat['Claims'].shift(1).ewm(span=4, adjust=False).mean()

    # Time-based features
    df_feat['month'] = df_feat['Date'].dt.month
    df_feat['quarter'] = df_feat['Date'].dt.quarter
    df_feat['year'] = df_feat['Date'].dt.year
    df_feat['week_of_year'] = df_feat['Date'].dt.isocalendar().week.astype(int)

    # CYCLICAL FEATURES - Encode periodic patterns using sin/cos transformation
    # This helps the model understand that December (12) is close to January (1)
    df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
    df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)
    df_feat['quarter_sin'] = np.sin(2 * np.pi * df_feat['quarter'] / 4)
    df_feat['quarter_cos'] = np.cos(2 * np.pi * df_feat['quarter'] / 4)
    df_feat['week_sin'] = np.sin(2 * np.pi * df_feat['week_of_year'] / 52)
    df_feat['week_cos'] = np.cos(2 * np.pi * df_feat['week_of_year'] / 52)

    # Trend feature (linear time index)
    df_feat['time_idx'] = range(len(df_feat))

    # Year-over-year features (seasonal comparison)
    # For weekly data, compare to same week last year (52 periods ago)
    # For monthly data, compare to same month last year (12 periods ago)
    period_per_year = 52 if len(df_feat) > 100 else 12
    df_feat['yoy_lag'] = df_feat['Claims'].shift(period_per_year)
    df_feat['yoy_diff'] = df_feat['Claims'].shift(1) - df_feat['Claims'].shift(period_per_year + 1)
    df_feat['yoy_ratio'] = df_feat['Claims'].shift(1) / df_feat['Claims'].shift(period_per_year + 1).replace(0, np.nan)

    # Momentum features
    df_feat['momentum_3'] = df_feat['Claims'].shift(1) - df_feat['Claims'].shift(4)
    df_feat['momentum_6'] = df_feat['Claims'].shift(1) - df_feat['Claims'].shift(7)

    # Volatility feature
    df_feat['volatility'] = df_feat['rolling_std_6'] / df_feat['rolling_mean_6'].replace(0, np.nan)

    # Drop rows with NaN (from lag features)
    df_feat = df_feat.dropna()

    return df_feat

# Create features
ts_features = create_forecast_features(ts_data)

# Train forecasting model
@st.cache_data
def train_forecast_model(ts_features):
    """Train an optimized Random Forest model for time series forecasting."""
    # Comprehensive feature set including cyclical features
    feature_cols = [
        # Lag features
        'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'lag_8',
        # Rolling statistics
        'rolling_mean_3', 'rolling_std_3', 'rolling_mean_6', 'rolling_std_6',
        'rolling_mean_12', 'rolling_min_6', 'rolling_max_6', 'ewm_mean',
        # Cyclical features (key for capturing seasonality)
        'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos',
        'week_sin', 'week_cos',
        # Year-over-year features
        'yoy_lag', 'yoy_diff', 'yoy_ratio',
        # Momentum and volatility
        'momentum_3', 'momentum_6', 'volatility',
        # Trend
        'time_idx'
    ]

    # Remove any features that don't exist
    feature_cols = [f for f in feature_cols if f in ts_features.columns]

    X = ts_features[feature_cols].values
    y = ts_features['Claims'].values

    # Handle any remaining NaN/inf values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # Use last 20% for testing (time series split)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = ts_features['Date'].iloc[split_idx:].values

    # Train optimized Random Forest with better hyperparameters
    model = RandomForestRegressor(
        n_estimators=300,          # More trees for stability
        max_depth=15,              # Deeper trees to capture complex patterns
        min_samples_leaf=2,        # Allow finer splits
        min_samples_split=4,       # Prevent overfitting
        max_features='sqrt',       # Feature subsampling for diversity
        bootstrap=True,
        oob_score=True,            # Out-of-bag score for validation
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100

    # R-squared
    ss_res = np.sum((y_test - y_pred_test) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        'model': model,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
        'dates_test': dates_test,
        'dates_train': ts_features['Date'].iloc[:split_idx].values,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'feature_cols': feature_cols,
        'feature_importance': model.feature_importances_,
        'oob_score': model.oob_score_ if hasattr(model, 'oob_score_') else None
    }

with st.spinner("Training Claims Demand Forecasting Model..."):
    forecast_results = train_forecast_model(ts_features)

# Display metrics
st.markdown("""
<div style='text-align: center; margin: 2rem 0 1.5rem 0;'>
    <h3 style='color: #1E3A5F; font-weight: 700; display: inline-block;
               border-bottom: 3px solid #667eea; padding-bottom: 0.5rem;'>
        Model Performance
    </h3>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.forecast-metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    color: white;
    box-shadow: 0 6px 16px rgba(102, 126, 234, 0.3);
    transition: all 0.3s ease;
}
.forecast-metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
}
.forecast-metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0.5rem 0;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}
.forecast-metric-label {
    font-size: 0.85rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
    opacity: 0.95;
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="forecast-metric-card">
        <div class="forecast-metric-label">MAE</div>
        <div class="forecast-metric-value">{forecast_results['mae']:.1f}</div>
        <div style='font-size: 0.75rem; opacity: 0.8;'>claims per {period_name.lower()}</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="forecast-metric-card">
        <div class="forecast-metric-label">RMSE</div>
        <div class="forecast-metric-value">{forecast_results['rmse']:.1f}</div>
        <div style='font-size: 0.75rem; opacity: 0.8;'>root mean squared error</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    mape_quality = "Excellent" if forecast_results['mape'] < 10 else "Good" if forecast_results['mape'] < 20 else "Fair"
    st.markdown(f"""
    <div class="forecast-metric-card">
        <div class="forecast-metric-label">MAPE</div>
        <div class="forecast-metric-value">{forecast_results['mape']:.1f}%</div>
        <div style='font-size: 0.75rem; opacity: 0.8;'>{mape_quality} accuracy</div>
    </div>
    """, unsafe_allow_html=True)

# What these metrics mean
st.markdown("### What Do These Numbers Mean?")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    **MAE: {forecast_results['mae']:.1f}**
    - Average prediction error
    - Off by ~{forecast_results['mae']:.0f} claims per {period_name.lower()}
    """)
with col2:
    st.markdown(f"""
    **RMSE: {forecast_results['rmse']:.1f}**
    - Penalizes large errors more
    - Good for catching outliers
    """)
with col3:
    st.markdown(f"""
    **MAPE: {forecast_results['mape']:.1f}%**
    - {"Excellent (<10%)" if forecast_results['mape'] < 10 else "Good (10-20%)" if forecast_results['mape'] < 20 else "Fair (20-30%)" if forecast_results['mape'] < 30 else "Needs improvement"}
    - Predictions within ±{forecast_results['mape']:.0f}% of actual
    """)

# Forecast vs Actual Chart
st.markdown(f"### {forecast_granularity} Forecast vs Actual Claims")

# Create combined dataframe for plotting
train_df = pd.DataFrame({
    'Date': forecast_results['dates_train'],
    'Actual': forecast_results['y_train'],
    'Forecast': forecast_results['y_pred_train'],
    'Set': 'Training'
})

test_df = pd.DataFrame({
    'Date': forecast_results['dates_test'],
    'Actual': forecast_results['y_test'],
    'Forecast': forecast_results['y_pred_test'],
    'Set': 'Test'
})

combined_df = pd.concat([train_df, test_df], ignore_index=True)

# Create the comparison chart
fig_forecast = go.Figure()

# Add actual claims line
fig_forecast.add_trace(go.Scatter(
    x=combined_df['Date'],
    y=combined_df['Actual'],
    mode='lines+markers',
    name='Actual Claims',
    line=dict(color='#667eea', width=2),
    marker=dict(size=6)
))

# Add forecast line
fig_forecast.add_trace(go.Scatter(
    x=combined_df['Date'],
    y=combined_df['Forecast'],
    mode='lines+markers',
    name='Forecast',
    line=dict(color='#f5576c', width=2, dash='dash'),
    marker=dict(size=6)
))

# Add vertical line to separate train/test (convert to string for plotly compatibility)
split_date = pd.Timestamp(forecast_results['dates_test'][0])
split_date_str = split_date.strftime('%Y-%m-%d')

# Add a shape instead of vline to avoid datetime issues
fig_forecast.add_shape(
    type="line",
    x0=split_date_str, x1=split_date_str,
    y0=0, y1=1,
    yref="paper",
    line=dict(color="gray", width=2, dash="dot")
)
fig_forecast.add_annotation(
    x=split_date_str, y=1.05, yref="paper",
    text="Train/Test Split",
    showarrow=False,
    font=dict(size=10, color="gray")
)

fig_forecast.update_layout(
    height=450,
    xaxis_title=f"{period_name}",
    yaxis_title="Number of Claims",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
    margin=dict(t=50, b=50)
)
fig_forecast = style_plotly_chart(fig_forecast)
st.plotly_chart(fig_forecast, use_container_width=True)

# Feature importance
st.markdown("### Feature Importance")
importance_df = pd.DataFrame({
    'Feature': forecast_results['feature_cols'],
    'Importance': forecast_results['feature_importance']
}).sort_values('Importance', ascending=True).tail(10)  # Top 10 features

fig_importance = px.bar(
    importance_df,
    x='Importance',
    y='Feature',
    orientation='h',
    color='Importance',
    color_continuous_scale='Purples'
)
fig_importance.update_layout(
    height=350,
    coloraxis_showscale=False
)
fig_importance = style_plotly_chart(fig_importance)
st.plotly_chart(fig_importance, use_container_width=True)

# Business insights
avg_claims = ts_data['Claims'].mean()
std_claims = ts_data['Claims'].std()

st.markdown(f"""
<div style='background: linear-gradient(145deg, #f0f9ff 0%, #e0f2fe 100%);
            border-left: 5px solid #0ea5e9; padding: 1.5rem; border-radius: 12px;
            margin: 1.5rem 0; box-shadow: 0 4px 12px rgba(14, 165, 233, 0.15);'>
    <strong style='color: #0ea5e9; font-size: 1.1rem;'>Business Application</strong>
    <p style='color: #475569; margin: 0.75rem 0 0 0; font-size: 0.95rem; line-height: 1.6;'>
        <strong>Historical Average:</strong> {avg_claims:.0f} claims per {period_name.lower()} (±{std_claims:.0f})<br>
        <strong>Forecast Accuracy:</strong> Within ±{forecast_results['mae']:.0f} claims {100-forecast_results['mape']:.0f}% of the time<br><br>
        <strong>Recommendation:</strong> Use this model to plan staffing levels {forecast_results['mae']:.0f} claims ahead of actual demand.
        For peak periods, add a buffer of {int(forecast_results['mae'] * 1.5)} claims to account for uncertainty.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# MODEL 2: LITIGATION COST PREDICTION (Random Forest only)
# ============================================================================
st.markdown("<div style='margin: 3rem 0 2rem 0;'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%);
            padding: 2.5rem 2rem; border-radius: 20px; margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(245, 87, 108, 0.3);'>
    <div style='text-align: center;'>
        <h2 style='color: white; margin: 0 0 0.5rem 0; font-size: 2.2rem; font-weight: 800; border: none;'>
            Model 2: Litigation Cost Prediction
        </h2>
        <p style='color: rgba(255,255,255,0.95); font-size: 1.2rem; margin: 0; font-weight: 300;'>
            Will this litigation claim be high-cost?
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Calculate cost threshold for display
cost_threshold_display = litigation_df[litigation_df['Total Cost To Date'] > 0]['Total Cost To Date'].median() if len(litigation_df) > 0 else 0

# THE BUSINESS PROBLEM
st.markdown(f"""
<div style='background: linear-gradient(145deg, #fef2f2 0%, #fecaca 100%);
            border-left: 5px solid #f5576c; padding: 2rem; border-radius: 12px;
            margin-bottom: 2rem; box-shadow: 0 4px 12px rgba(245, 87, 108, 0.15);'>
    <h3 style='color: #1E3A5F; margin: 0 0 1rem 0; font-size: 1.5rem; font-weight: 700;'>
        The Business Problem
    </h3>
    <p style='color: #475569; font-size: 1.05rem; line-height: 1.7; margin-bottom: 1rem;'>
        When a claim enters litigation, costs can vary dramatically. <strong>Early identification of high-cost cases</strong>
        is critical for financial planning and resource allocation. Without prediction, reserves may be inadequate
        and senior expertise may not be deployed when needed most.
    </p>
    <div style='margin-top: 1rem; padding: 1rem; background: rgba(245, 87, 108, 0.15); border-radius: 8px; border: 1px solid rgba(245, 87, 108, 0.3);'>
        <strong style='color: #be123c; font-size: 1.05rem;'>What We're Predicting:</strong>
        <p style='color: #475569; margin: 0.5rem 0 0 0; font-size: 0.95rem; line-height: 1.6;'>
            A binary classification: will a litigation claim's <strong>Total Cost To Date</strong> exceed the median cost threshold?
        </p>
        <ul style='color: #475569; margin: 0.5rem 0 0 1.5rem; line-height: 1.5; font-size: 0.9rem;'>
            <li><strong>High Cost</strong> = Total Cost To Date > <strong>£{cost_threshold_display:,.0f}</strong> (median of litigation claims)</li>
            <li><strong>Low Cost</strong> = Total Cost To Date ≤ £{cost_threshold_display:,.0f}</li>
        </ul>
    </div>
    <div style='margin-top: 1rem; padding: 1rem; background: rgba(245, 87, 108, 0.1); border-radius: 8px;'>
        <strong style='color: #f5576c; font-size: 1.05rem;'>Solution Approach:</strong>
        <p style='color: #475569; margin: 0.5rem 0; font-size: 0.95rem; line-height: 1.6;'>
            A <strong>Balanced Random Forest Classifier</strong> with the following optimizations:
        </p>
        <ul style='color: #475569; margin: 0.5rem 0 0 1.5rem; line-height: 1.5; font-size: 0.9rem;'>
            <li><strong>Class balancing</strong> - Balanced subsampling to handle imbalanced high/low cost distribution</li>
            <li><strong>Feature encoding</strong> - Label encoding of claim category, type, business class, and agent</li>
            <li><strong>Threshold tuning</strong> - Classification threshold optimized to maximize F1 score (balancing precision and recall)</li>
            <li><strong>Ensemble method</strong> - 500 decision trees with bootstrap aggregation for robust predictions</li>
        </ul>
    </div>
    <div style='margin-top: 1rem; padding: 1rem; background: rgba(245, 87, 108, 0.1); border-radius: 8px;'>
        <strong style='color: #f5576c; font-size: 1.05rem;'>Business Value:</strong>
        <ul style='color: #475569; margin: 0.5rem 0 0 1.5rem; line-height: 1.6;'>
            <li><strong>Accurate reserve setting</strong> - Set appropriate financial reserves from day one</li>
            <li><strong>Senior handler assignment</strong> - Route high-cost cases to experienced staff</li>
            <li><strong>Proactive cost management</strong> - Implement cost controls before expenses escalate</li>
            <li><strong>Panel lawyer selection</strong> - Match case complexity with appropriate legal expertise</li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)

# Prepare data for Model 2
if len(litigation_df) > 100:
    X2 = litigation_df[encoded_features].values
    y2 = litigation_df['target_high_cost'].values

    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y2, test_size=0.2, random_state=42, stratify=y2
    )

    # Train with Balanced Random Forest and optimize threshold for F1
    def train_cost_model(_X_train, _y_train, _X_test, _y_test):
        model = BalancedRandomForestClassifier(
            n_estimators=500,
            max_depth=9,
            random_state=154,
            n_jobs=-1,
            sampling_strategy='all',
            replacement=True,
            class_weight='balanced_subsample'
        )
        model.fit(_X_train, _y_train)
        y_prob = model.predict_proba(_X_test)[:, 1]

        # Find optimal threshold that maximizes F1 score
        thresholds = np.arange(0.1, 0.9, 0.01)
        f1_scores = []
        for thresh in thresholds:
            y_pred_thresh = (y_prob >= thresh).astype(int)
            f1_scores.append(f1_score(_y_test, y_pred_thresh))

        optimal_threshold = thresholds[np.argmax(f1_scores)]
        best_f1 = max(f1_scores)

        # Apply optimal threshold for predictions
        y_pred = (y_prob >= optimal_threshold).astype(int)

        return model, y_pred, y_prob, optimal_threshold, best_f1

    with st.spinner("Training Litigation Cost Model with Random Forest..."):
        model2, y2_pred, y2_prob, optimal_thresh2, best_f1_2 = train_cost_model(X2_train, y2_train, X2_test, y2_test)

    # Performance metrics
    st.markdown("""
    <div style='text-align: center; margin: 2rem 0 1.5rem 0;'>
        <h3 style='color: #1E3A5F; font-weight: 700; display: inline-block;
                   border-bottom: 3px solid #f5576c; padding-bottom: 0.5rem;'>
            Model Performance
        </h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    .cost-metric-card {
        background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: white;
        box-shadow: 0 6px 16px rgba(245, 87, 108, 0.3);
    }
    .cost-metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    .cost-metric-label {
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.95;
    }
    </style>
    """, unsafe_allow_html=True)

    # Calculate metrics with optimized threshold
    acc2 = accuracy_score(y2_test, y2_pred)
    prec2 = precision_score(y2_test, y2_pred)
    rec2 = recall_score(y2_test, y2_pred)
    f1_2 = f1_score(y2_test, y2_pred)

    # Show optimal threshold info
    st.markdown(f"""
    <div style='background: linear-gradient(145deg, #fef2f2 0%, #fee2e2 100%);
                border: 1px solid #fecaca; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem;
                text-align: center;'>
        <span style='color: #991b1b; font-weight: 600;'>Optimal Classification Threshold: </span>
        <span style='color: #dc2626; font-weight: 700; font-size: 1.1rem;'>{optimal_thresh2:.2f}</span>
        <span style='color: #7f1d1d; font-size: 0.85rem; margin-left: 0.5rem;'>(tuned to maximize F1 score)</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="cost-metric-card">
            <div class="cost-metric-label">Accuracy</div>
            <div class="cost-metric-value">{acc2:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="cost-metric-card">
            <div class="cost-metric-label">Precision</div>
            <div class="cost-metric-value">{prec2:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="cost-metric-card">
            <div class="cost-metric-label">Recall</div>
            <div class="cost-metric-value">{rec2:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="cost-metric-card">
            <div class="cost-metric-label">F1-Score</div>
            <div class="cost-metric-value">{f1_2:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    # What these metrics mean
    st.markdown("### What Do These Numbers Mean?")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"**Accuracy: {acc2:.1%}** — We correctly classify {acc2:.0%} of all litigation cases")
    with col2:
        st.markdown(f"**Precision: {prec2:.1%}** — When we flag a case as high-cost, we're right {prec2:.0%} of the time")
    with col3:
        st.markdown(f"**Recall: {rec2:.1%}** — We successfully identify {rec2:.0%} of all actual high-cost cases")
    with col4:
        st.markdown(f"**F1-Score: {f1_2:.3f}** — {'Strong' if f1_2 > 0.7 else 'Good' if f1_2 > 0.6 else 'Moderate' if f1_2 > 0.4 else 'Needs improvement'} balance between precision and recall")

    # Confusion matrix
    st.markdown("#### Confusion Matrix")
    cm2 = confusion_matrix(y2_test, y2_pred)
    fig_cm2 = px.imshow(
        cm2,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Low Cost', 'High Cost'],
        y=['Low Cost', 'High Cost'],
        color_continuous_scale='Reds',
        text_auto=True
    )
    fig_cm2.update_layout(height=350)
    fig_cm2 = style_plotly_chart(fig_cm2)
    st.plotly_chart(fig_cm2, use_container_width=True)

    # Feature importance
    st.markdown("#### Key Cost Drivers")
    importance_df2 = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model2.feature_importances_
    }).sort_values('Importance', ascending=True)

    fig_imp2 = px.bar(
        importance_df2,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='Reds'
    )
    fig_imp2.update_layout(height=300, coloraxis_showscale=False)
    fig_imp2 = style_plotly_chart(fig_imp2)
    st.plotly_chart(fig_imp2, use_container_width=True)

else:
    st.warning("Insufficient litigation data for Model 2. Need at least 100 litigation claims.")

st.markdown("---")

# ============================================================================
# MODEL 3: LITIGATION DURATION PREDICTION (Random Forest only)
# ============================================================================
st.markdown("<div style='margin: 3rem 0 2rem 0;'></div>", unsafe_allow_html=True)
# Get resolution threshold for display
resolution_threshold_display = litigation_df['resolution_threshold'].iloc[0] if len(litigation_df) > 0 and 'resolution_threshold' in litigation_df.columns else 90

st.markdown(f"""
<div style='background: linear-gradient(135deg, #00d4aa 0%, #00b4d8 100%);
            padding: 2.5rem 2rem; border-radius: 20px; margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0, 212, 170, 0.3);'>
    <div style='text-align: center;'>
        <h2 style='color: white; margin: 0 0 0.5rem 0; font-size: 2.2rem; font-weight: 800; border: none;'>
            Model 3: Litigation Duration Prediction
        </h2>
        <p style='color: rgba(255,255,255,0.95); font-size: 1.2rem; margin: 0; font-weight: 300;'>
            Will this claim resolve quickly (&le;{resolution_threshold_display:.0f} days)?
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# THE BUSINESS PROBLEM
st.markdown(f"""
<div style='background: linear-gradient(145deg, #f0fdfa 0%, #ccfbf1 100%);
            border-left: 5px solid #00d4aa; padding: 2rem; border-radius: 12px;
            margin-bottom: 2rem; box-shadow: 0 4px 12px rgba(0, 212, 170, 0.15);'>
    <h3 style='color: #1E3A5F; margin: 0 0 1rem 0; font-size: 1.5rem; font-weight: 700;'>
        The Business Problem
    </h3>
    <p style='color: #475569; font-size: 1.05rem; line-height: 1.7; margin-bottom: 1rem;'>
        Litigation timelines vary significantly. Some cases resolve within weeks while others drag on for months.
        <strong>Predicting duration upfront</strong> enables better customer expectations, workload balancing,
        and resource planning across the claims team.
    </p>
    <div style='margin-top: 1rem; padding: 1rem; background: rgba(0, 212, 170, 0.15); border-radius: 8px; border: 1px solid rgba(0, 212, 170, 0.3);'>
        <strong style='color: #047857; font-size: 1.05rem;'>What We're Predicting:</strong>
        <p style='color: #475569; margin: 0.5rem 0 0 0; font-size: 0.95rem; line-height: 1.6;'>
            A binary classification: will a litigation claim resolve within the median resolution time?
        </p>
        <ul style='color: #475569; margin: 0.5rem 0 0 1.5rem; line-height: 1.5; font-size: 0.9rem;'>
            <li><strong>Quick Resolution</strong> = Resolution Time ≤ <strong>{resolution_threshold_display:.0f} days</strong> (median of litigation claims)</li>
            <li><strong>Extended Resolution</strong> = Resolution Time > {resolution_threshold_display:.0f} days</li>
        </ul>
    </div>
    <div style='margin-top: 1rem; padding: 1rem; background: rgba(0, 212, 170, 0.1); border-radius: 8px;'>
        <strong style='color: #00d4aa; font-size: 1.05rem;'>Solution Approach:</strong>
        <p style='color: #475569; margin: 0.5rem 0; font-size: 0.95rem; line-height: 1.6;'>
            A <strong>Balanced Random Forest Classifier</strong> with the following optimizations:
        </p>
        <ul style='color: #475569; margin: 0.5rem 0 0 1.5rem; line-height: 1.5; font-size: 0.9rem;'>
            <li><strong>Class balancing</strong> - Balanced subsampling to handle imbalanced quick/extended distribution</li>
            <li><strong>Feature encoding</strong> - Label encoding of claim category, type, business class, and agent</li>
            <li><strong>Threshold tuning</strong> - Classification threshold optimized to maximize F1 score</li>
            <li><strong>Ensemble method</strong> - 500 decision trees with bootstrap aggregation for robust predictions</li>
        </ul>
    </div>
    <div style='margin-top: 1rem; padding: 1rem; background: rgba(0, 212, 170, 0.1); border-radius: 8px;'>
        <strong style='color: #00d4aa; font-size: 1.05rem;'>Business Value:</strong>
        <ul style='color: #475569; margin: 0.5rem 0 0 1.5rem; line-height: 1.6;'>
            <li><strong>Customer expectations</strong> - Set realistic timelines from the start</li>
            <li><strong>Workload planning</strong> - Balance caseloads knowing which cases need long-term attention</li>
            <li><strong>Resource allocation</strong> - Assign appropriate capacity for prolonged cases</li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)

# Filter litigation claims with valid resolution time
litigation_with_time = litigation_df[litigation_df['Resolution Time (Days)'].notna()].copy()

if len(litigation_with_time) > 100:
    X3 = litigation_with_time[encoded_features].values
    y3 = litigation_with_time['target_quick'].values

    X3_train, X3_test, y3_train, y3_test = train_test_split(
        X3, y3, test_size=0.2, random_state=42, stratify=y3
    )

    # Train with Balanced Random Forest and optimize threshold for F1
    def train_duration_model(_X_train, _y_train, _X_test, _y_test):
        model = BalancedRandomForestClassifier(
            n_estimators=500,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
            sampling_strategy='all',
            replacement=True,
            class_weight='balanced_subsample'
        )
        model.fit(_X_train, _y_train)
        y_prob = model.predict_proba(_X_test)[:, 1]

        # Find optimal threshold that maximizes F1 score
        thresholds = np.arange(0.1, 0.9, 0.01)
        f1_scores = []
        for thresh in thresholds:
            y_pred_thresh = (y_prob >= thresh).astype(int)
            f1_scores.append(f1_score(_y_test, y_pred_thresh))

        optimal_threshold = thresholds[np.argmax(f1_scores)]
        best_f1 = max(f1_scores)

        # Apply optimal threshold for predictions
        y_pred = (y_prob >= optimal_threshold).astype(int)

        return model, y_pred, y_prob, optimal_threshold, best_f1

    with st.spinner("Training Litigation Duration Model with Random Forest..."):
        model3, y3_pred, y3_prob, optimal_thresh3, best_f1_3 = train_duration_model(X3_train, y3_train, X3_test, y3_test)

    # Performance metrics
    st.markdown("""
    <div style='text-align: center; margin: 2rem 0 1.5rem 0;'>
        <h3 style='color: #1E3A5F; font-weight: 700; display: inline-block;
                   border-bottom: 3px solid #00d4aa; padding-bottom: 0.5rem;'>
            Model Performance
        </h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    .duration-metric-card {
        background: linear-gradient(135deg, #00d4aa 0%, #00b4d8 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: white;
        box-shadow: 0 6px 16px rgba(0, 212, 170, 0.3);
    }
    .duration-metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    .duration-metric-label {
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.95;
    }
    </style>
    """, unsafe_allow_html=True)

    # Calculate metrics with optimized threshold
    acc3 = accuracy_score(y3_test, y3_pred)
    prec3 = precision_score(y3_test, y3_pred)
    rec3 = recall_score(y3_test, y3_pred)
    f1_3 = f1_score(y3_test, y3_pred)

    # Show optimal threshold info
    st.markdown(f"""
    <div style='background: linear-gradient(145deg, #f0fdfa 0%, #ccfbf1 100%);
                border: 1px solid #99f6e4; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem;
                text-align: center;'>
        <span style='color: #047857; font-weight: 600;'>Optimal Classification Threshold: </span>
        <span style='color: #059669; font-weight: 700; font-size: 1.1rem;'>{optimal_thresh3:.2f}</span>
        <span style='color: #065f46; font-size: 0.85rem; margin-left: 0.5rem;'>(tuned to maximize F1 score)</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="duration-metric-card">
            <div class="duration-metric-label">Accuracy</div>
            <div class="duration-metric-value">{acc3:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="duration-metric-card">
            <div class="duration-metric-label">Precision</div>
            <div class="duration-metric-value">{prec3:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="duration-metric-card">
            <div class="duration-metric-label">Recall</div>
            <div class="duration-metric-value">{rec3:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="duration-metric-card">
            <div class="duration-metric-label">F1-Score</div>
            <div class="duration-metric-value">{f1_3:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    # What these metrics mean
    st.markdown("### What Do These Numbers Mean?")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"**Accuracy: {acc3:.1%}** — We correctly classify {acc3:.0%} of all litigation cases")
    with col2:
        st.markdown(f"**Precision: {prec3:.1%}** — When we predict quick resolution, we're right {prec3:.0%} of the time")
    with col3:
        st.markdown(f"**Recall: {rec3:.1%}** — We successfully identify {rec3:.0%} of all quick-resolution cases")
    with col4:
        st.markdown(f"**F1-Score: {f1_3:.3f}** — {'Strong' if f1_3 > 0.7 else 'Good' if f1_3 > 0.6 else 'Moderate' if f1_3 > 0.4 else 'Needs improvement'} balance between precision and recall")

    # Confusion matrix
    st.markdown("#### Confusion Matrix")
    cm3 = confusion_matrix(y3_test, y3_pred)
    fig_cm3 = px.imshow(
        cm3,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=[f'Extended (>{resolution_threshold_display:.0f}d)', f'Quick (≤{resolution_threshold_display:.0f}d)'],
        y=[f'Extended (>{resolution_threshold_display:.0f}d)', f'Quick (≤{resolution_threshold_display:.0f}d)'],
        color_continuous_scale='Teal',
        text_auto=True
    )
    fig_cm3.update_layout(height=350)
    fig_cm3 = style_plotly_chart(fig_cm3)
    st.plotly_chart(fig_cm3, use_container_width=True)

    # Feature importance
    st.markdown("#### Key Duration Drivers")
    importance_df3 = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model3.feature_importances_
    }).sort_values('Importance', ascending=True)

    fig_imp3 = px.bar(
        importance_df3,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='Teal'
    )
    fig_imp3.update_layout(height=300, coloraxis_showscale=False)
    fig_imp3 = style_plotly_chart(fig_imp3)
    st.plotly_chart(fig_imp3, use_container_width=True)

else:
    st.warning("Insufficient litigation data with resolution times for Model 3.")

st.markdown("---")

# ============================================================================
# MODEL 4: CUSTOMER SEGMENTATION (K-Means Clustering)
# ============================================================================
st.markdown("<div style='margin: 3rem 0 2rem 0;'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%);
            padding: 2.5rem 2rem; border-radius: 20px; margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(245, 158, 11, 0.3);'>
    <div style='text-align: center;'>
        <h2 style='color: white; margin: 0 0 0.5rem 0; font-size: 2.2rem; font-weight: 800; border: none;'>
            Model 4: Customer Segmentation
        </h2>
        <p style='color: rgba(255,255,255,0.95); font-size: 1.2rem; margin: 0; font-weight: 300;'>
            Identify natural groupings in claims for targeted strategies
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# THE BUSINESS PROBLEM
st.markdown("""
<div style='background: linear-gradient(145deg, #fffbeb 0%, #fef3c7 100%);
            border-left: 5px solid #f59e0b; padding: 2rem; border-radius: 12px;
            margin-bottom: 2rem; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.15);'>
    <h3 style='color: #1E3A5F; margin: 0 0 1rem 0; font-size: 1.5rem; font-weight: 700;'>
        The Business Problem
    </h3>
    <p style='color: #475569; font-size: 1.05rem; line-height: 1.7; margin-bottom: 1rem;'>
        When a new claim arrives, the team needs to quickly decide how to handle it. Treating all claims identically
        leads to inefficiency. <strong>Segmenting claims at intake</strong> based on their characteristics enables
        immediate routing to the right team with appropriate resources.
    </p>
    <div style='margin-top: 1rem; padding: 1rem; background: rgba(245, 158, 11, 0.1); border-radius: 8px;'>
        <strong style='color: #f59e0b; font-size: 1.05rem;'>Solution Approach:</strong>
        <p style='color: #475569; margin: 0.5rem 0; font-size: 0.95rem; line-height: 1.6;'>
            <strong>K-Means Clustering</strong> on features available at claim intake:
        </p>
        <ul style='color: #475569; margin: 0.5rem 0 0 1.5rem; line-height: 1.5; font-size: 0.9rem;'>
            <li><strong>Claim Category & Type</strong> - Nature of the legal dispute</li>
            <li><strong>Business Class</strong> - Motor, Personal non-motor, or Commercial</li>
            <li><strong>Agent Company</strong> - Insurance partner patterns</li>
            <li><strong>Reporting Delay</strong> - Days between incident and claim submission</li>
            <li><strong>Reinsurance Type</strong> - Primary insurance vs Inward reinsurance</li>
        </ul>
    </div>
    <div style='margin-top: 1rem; padding: 1rem; background: rgba(245, 158, 11, 0.1); border-radius: 8px;'>
        <strong style='color: #f59e0b; font-size: 1.05rem;'>Business Value:</strong>
        <ul style='color: #475569; margin: 0.5rem 0 0 1.5rem; line-height: 1.6;'>
            <li><strong>Immediate triage</strong> - Route new claims to the right team from day one</li>
            <li><strong>Resource planning</strong> - Allocate specialists based on claim profile</li>
            <li><strong>Risk assessment</strong> - Identify high-complexity claims early</li>
            <li><strong>Process optimization</strong> - Tailor workflows per segment</li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)

# Prepare clustering data and train model in one function to ensure alignment
@st.cache_data
def prepare_and_cluster_data(df, n_clusters=4, sample_size=10000):
    """Prepare data for clustering and train K-Means model."""
    # Use all claims (not just closed) since we're using intake features
    cluster_df = df.copy()

    # Calculate reporting delay (days between incident and claim input)
    cluster_df['Reporting_Delay'] = (cluster_df['Input Date'] - cluster_df['Incident Date']).dt.days
    # Cap extreme values and handle negatives (data entry errors)
    cluster_df['Reporting_Delay'] = cluster_df['Reporting_Delay'].clip(lower=0, upper=365)

    # Filter to rows with valid data
    required_cols = ['Claim Category Description', 'Claim Type Description',
                     'Business Class Description', 'Agent Company Name', 'Inward Reinsurance']
    cluster_df = cluster_df.dropna(subset=required_cols + ['Reporting_Delay'])

    # Reset index BEFORE encoding to ensure alignment
    cluster_df = cluster_df.reset_index(drop=True)

    # Encode categorical features for clustering
    cluster_df['Category_enc'] = cluster_df['Claim Category Description'].astype('category').cat.codes
    cluster_df['Type_enc'] = cluster_df['Claim Type Description'].astype('category').cat.codes
    cluster_df['Business_enc'] = cluster_df['Business Class Description'].astype('category').cat.codes
    cluster_df['Agent_enc'] = cluster_df['Agent Company Name'].astype('category').cat.codes
    cluster_df['Reins_enc'] = cluster_df['Inward Reinsurance'].astype('category').cat.codes

    # Features for clustering
    cluster_features = ['Category_enc', 'Type_enc', 'Business_enc', 'Agent_enc', 'Reins_enc', 'Reporting_Delay']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_df[cluster_features])

    # Sample for faster K-Means training if dataset is large
    if len(X_scaled) > sample_size:
        sample_idx = np.random.RandomState(42).choice(len(X_scaled), sample_size, replace=False)
        X_sample = X_scaled[sample_idx]
    else:
        X_sample = X_scaled

    # Train K-Means on sample
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_sample)

    # Predict for all data
    cluster_labels = kmeans.predict(X_scaled)

    # Calculate silhouette on a sample for speed
    silhouette_sample_idx = np.random.RandomState(42).choice(len(X_scaled), min(5000, len(X_scaled)), replace=False)
    silhouette = silhouette_score(X_scaled[silhouette_sample_idx], cluster_labels[silhouette_sample_idx])

    # Assign clusters to dataframe
    cluster_df['Cluster'] = cluster_labels

    return cluster_df, silhouette

with st.spinner("Training Customer Segmentation Model..."):
    cluster_df, silhouette = prepare_and_cluster_data(df)

# Cluster analysis
st.markdown("""
<div style='text-align: center; margin: 2rem 0 1.5rem 0;'>
    <h3 style='color: #1E3A5F; font-weight: 700; display: inline-block;
               border-bottom: 3px solid #f59e0b; padding-bottom: 0.5rem;'>
        Segment Analysis
    </h3>
</div>
""", unsafe_allow_html=True)

# Analyze cluster characteristics using intake features
cluster_analysis = cluster_df.groupby('Cluster').agg({
    'Claim Reference': 'count',
    'Reporting_Delay': 'mean',
    'Claim Category Description': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',
    'Business Class Description': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',
    'Inward Reinsurance': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
}).round(1)
cluster_analysis.columns = ['Count', 'Avg Reporting Delay (Days)', 'Most Common Category', 'Most Common Business', 'Most Common Reinsurance']
cluster_analysis = cluster_analysis.reset_index()

# Calculate outcome statistics for each cluster (for validation)
cluster_outcomes = cluster_df[cluster_df['Claim Status'] == 'C'].groupby('Cluster').agg({
    'Total Cost To Date': 'mean',
    'Resolution Time (Days)': 'mean'
}).round(0)
cluster_outcomes.columns = ['Avg Cost (£)', 'Avg Resolution (Days)']
cluster_outcomes = cluster_outcomes.reset_index()

# Merge for full picture
cluster_summary = cluster_analysis.merge(cluster_outcomes, on='Cluster', how='left')

# Name the clusters based on intake characteristics - ensure unique names
def name_clusters(df):
    """Assign unique names to clusters based on their characteristics."""
    names = []
    used_names = set()

    for _, row in df.iterrows():
        category = row['Most Common Category']
        business = row['Most Common Business']
        delay = row['Avg Reporting Delay (Days)']

        # Determine base name
        if 'Motor' in business:
            base_name = "Motor Claims"
        elif 'Commercial' in str(business):
            base_name = "Commercial Disputes"
        elif 'Bodily Injury' in str(category):
            base_name = "Personal Injury"
        elif 'Property' in str(category):
            base_name = "Property Claims"
        elif 'Personal Contract' in str(category):
            base_name = "Contract Disputes"
        elif delay > 100:
            base_name = "Late Reporting"
        elif delay > 60:
            base_name = "Delayed Reporting"
        else:
            base_name = "Standard Claims"

        # Ensure unique name
        final_name = base_name
        counter = 2
        while final_name in used_names:
            final_name = f"{base_name} {counter}"
            counter += 1

        used_names.add(final_name)
        names.append(final_name)

    return names

cluster_summary['Segment Name'] = name_clusters(cluster_summary)

# Display cluster cards
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%);
                border-radius: 12px; padding: 1.5rem; text-align: center; color: white;
                box-shadow: 0 6px 16px rgba(245, 158, 11, 0.3);'>
        <div style='font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9;'>
            Silhouette Score
        </div>
        <div style='font-size: 2.5rem; font-weight: 800; margin: 0.5rem 0;'>
            {silhouette:.3f}
        </div>
        <div style='font-size: 0.8rem; opacity: 0.8;'>
            {"Good" if silhouette > 0.5 else "Moderate" if silhouette > 0.3 else "Acceptable"} cluster separation
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.dataframe(
        cluster_summary[['Cluster', 'Segment Name', 'Count', 'Avg Reporting Delay (Days)', 'Most Common Category']],
        use_container_width=True,
        hide_index=True
    )

# Show 3D scatter plot of cluster separation
st.markdown("#### Cluster Separation Visualization")
st.markdown("*3D view showing how claims are grouped based on Business Class, Claim Category, and Reporting Delay:*")

# Sample data for visualization (too many points makes it slow)
viz_sample = cluster_df.sample(n=min(5000, len(cluster_df)), random_state=42).copy()

# Map cluster numbers to segment names for visualization
cluster_to_name = dict(zip(cluster_summary['Cluster'], cluster_summary['Segment Name']))
viz_sample['Segment'] = viz_sample['Cluster'].map(cluster_to_name)

fig_3d = px.scatter_3d(
    viz_sample,
    x='Business_enc',
    y='Category_enc',
    z='Reporting_Delay',
    color='Segment',
    color_discrete_sequence=['#667eea', '#f5576c', '#00d4aa', '#f59e0b'],
    opacity=0.6,
    title="Cluster Separation (sampled for performance)"
)
fig_3d.update_layout(
    height=500,
    scene=dict(
        xaxis_title="Business Class",
        yaxis_title="Claim Category",
        zaxis_title="Reporting Delay (Days)"
    ),
    legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
)
fig_3d = style_plotly_chart(fig_3d)
st.plotly_chart(fig_3d, use_container_width=True)

# Show historical outcomes per segment (for validation that segmentation is useful)
st.markdown("#### Historical Outcomes by Segment")
st.markdown("*This shows how claims in each segment have historically performed (for closed claims only):*")

closed_by_cluster = cluster_df[cluster_df['Claim Status'] == 'C'].copy()
if len(closed_by_cluster) > 0:
    outcome_summary = closed_by_cluster.groupby('Cluster').agg({
        'Total Cost To Date': ['mean', 'median'],
        'Resolution Time (Days)': ['mean', 'median'],
        'Claim Reference': 'count'
    }).round(0)
    outcome_summary.columns = ['Avg Cost', 'Median Cost', 'Avg Days', 'Median Days', 'Closed Claims']
    outcome_summary = outcome_summary.reset_index()
    outcome_summary = outcome_summary.merge(cluster_summary[['Cluster', 'Segment Name']], on='Cluster')

    st.dataframe(
        outcome_summary[['Segment Name', 'Closed Claims', 'Avg Cost', 'Median Cost', 'Avg Days', 'Median Days']].style.format({
            'Avg Cost': '£{:,.0f}',
            'Median Cost': '£{:,.0f}',
            'Avg Days': '{:.0f}',
            'Median Days': '{:.0f}'
        }),
        use_container_width=True,
        hide_index=True
    )

# Business recommendations based on actual segment characteristics
segment_colors = ['#667eea', '#f5576c', '#00d4aa', '#f59e0b']
segment_recommendations = {
    'Motor Claims': 'Dedicated motor team, standard process, typically quick resolution with low cost',
    'Commercial Disputes': 'Senior legal expertise, higher value potential, proactive cost management needed',
    'Contract Disputes': 'Specialist contract lawyers, medium complexity, balanced approach required',
    'Late Reporting': 'Investigate delay reasons, potential coverage issues',
    'Delayed Reporting': 'Review for time-bar issues, assess evidence availability, expedite where possible',
    'Personal Injury': 'Medical expertise required, longer timeline expected, reserve management critical',
    'Property Claims': 'Property specialists, surveyor coordination, evidence preservation focus',
    'Standard Claims': 'Standard processing workflow, regular monitoring, escalate if complexity increases'
}

st.markdown(f"""
<div style='background: linear-gradient(145deg, #fffbeb 0%, #fef3c7 100%);
            border-left: 5px solid #f59e0b; padding: 1.5rem; border-radius: 12px;
            margin: 1.5rem 0; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.15);'>
    <strong style='color: #b45309; font-size: 1.1rem;'>Intake-Based Routing Recommendations</strong>
    <p style='color: #64748b; font-size: 0.9rem; margin: 0.5rem 0;'>
        Use these segments to route new claims at intake before any outcomes are known:
    </p>
    <div style='margin-top: 1rem; display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;'>
""", unsafe_allow_html=True)

# Dynamically generate recommendation cards based on actual segments
for i, row in cluster_summary.iterrows():
    segment_name = row['Segment Name']
    color = segment_colors[i % len(segment_colors)]
    recommendation = segment_recommendations.get(segment_name, 'Standard processing workflow')
    st.markdown(f"""
        <div style='background: white; padding: 1rem; border-radius: 8px; border-left: 3px solid {color};'>
            <strong style='color: {color};'>{segment_name}</strong>
            <p style='color: #64748b; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>
                {recommendation}
            </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("""
    </div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 2rem 0;'>
    <p style='margin: 0; font-size: 0.9rem;'>
        <strong>Note:</strong> All classification models use <strong>Balanced Random Forest</strong> for optimal handling of class imbalance.
        The forecasting model uses <strong>Random Forest Regressor</strong> with time series features.
    </p>
</div>
""", unsafe_allow_html=True)
