"""
Data Loading and Preprocessing Module for ARAG Claims Analysis
"""
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
from pathlib import Path


@st.cache_data
def load_data(filepath: str = None) -> pd.DataFrame:
    # Get the directory where this file is located
    if filepath is None:
        base_dir = Path(__file__).parent.parent
        filepath = base_dir / "Data Science Sample data.csv"
    """Load and preprocess the ARAG claims dataset."""
    # Try different encodings to handle special characters like £
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(filepath, encoding='latin-1')
        except:
            df = pd.read_csv(filepath, encoding='cp1252')

    # Convert date columns
    date_columns = ['Incident Date', 'Input Date', 'First Validation Date', 'Closure Date']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce')

    # Calculate derived metrics
    df['Processing Time (Days)'] = (df['First Validation Date'] - df['Input Date']).dt.days
    df['Resolution Time (Days)'] = (df['Closure Date'] - df['Input Date']).dt.days
    df['Incident to Input (Days)'] = (df['Input Date'] - df['Incident Date']).dt.days

    # Extract temporal features
    df['Input Year'] = df['Input Date'].dt.year
    df['Input Month'] = df['Input Date'].dt.month
    df['Input Quarter'] = df['Input Date'].dt.quarter
    df['Input Day of Week'] = df['Input Date'].dt.dayofweek
    df['Closure Year'] = df['Closure Date'].dt.year

    # Create cost brackets
    df['Cost Bracket'] = pd.cut(
        df['Total Cost To Date'],
        bins=[-1, 0, 500, 1500, 5000, 20000, float('inf')],
        labels=['Zero Cost', 'Low (<£500)', 'Medium (£500-1.5K)',
                'High (£1.5K-5K)', 'Very High (£5K-20K)', 'Extreme (>£20K)']
    )

    # Flag for high-cost claims
    df['Is High Cost'] = df['Total Cost To Date'] > df['Total Cost To Date'].quantile(0.75)

    # Claim coverage status: whether claim was accepted for coverage (Closed/Open) vs rejected (Declined/Non-Policy)
    df['Claim Covered'] = df['Claim Status'].isin(['C', 'O'])
    # Keep 'Claim Successful' for backward compatibility but it means "covered/accepted"
    df['Claim Successful'] = df['Claim Covered']

    return df


def get_summary_stats(df: pd.DataFrame) -> dict:
    """Generate summary statistics for the dashboard."""
    closed_claims = df[df['Claim Status'] == 'C']

    return {
        'total_claims': len(df),
        'total_cost': df['Total Cost To Date'].sum(),
        'avg_cost': df['Total Cost To Date'].mean(),
        'median_cost': df['Total Cost To Date'].median(),
        'closed_claims': len(closed_claims),
        'open_claims': len(df[df['Claim Status'] == 'O']),
        'declined_claims': len(df[df['Claim Status'] == 'D']),
        'avg_resolution_time': closed_claims['Resolution Time (Days)'].mean(),
        'median_resolution_time': closed_claims['Resolution Time (Days)'].median(),
        'claim_coverage_rate': (df['Claim Status'].isin(['C', 'O']).sum() / len(df)) * 100,
        'unique_agents': df['Agent Company Name'].nunique(),
        'unique_claim_types': df['Claim Type Description'].nunique()
    }


def get_time_series_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare time series data for trend analysis."""
    monthly_data = df.groupby(df['Input Date'].dt.to_period('M')).agg({
        'Claim Reference': 'count',
        'Total Cost To Date': ['sum', 'mean'],
        'Resolution Time (Days)': 'mean'
    }).reset_index()

    monthly_data.columns = ['Period', 'Claim Count', 'Total Cost', 'Avg Cost', 'Avg Resolution Time']
    monthly_data['Period'] = monthly_data['Period'].astype(str)

    return monthly_data


def prepare_ml_features(df: pd.DataFrame) -> tuple:
    """Prepare features for machine learning models."""
    # Filter for relevant records
    ml_df = df[df['Claim Status'].isin(['C', 'D', 'N'])].copy()

    # Create target variable (1 = Closed successfully, 0 = Declined/Non-Policy)
    ml_df['Target'] = (ml_df['Claim Status'] == 'C').astype(int)

    # Feature engineering
    feature_cols = ['Claim Category Description', 'Claim Type Description',
                    'Business Class Description', 'Agent Company Name',
                    'Litigator Type Description', 'Litigator Panel Type',
                    'Inward Reinsurance', 'Processing Time (Days)',
                    'Incident to Input (Days)']

    return ml_df, feature_cols
