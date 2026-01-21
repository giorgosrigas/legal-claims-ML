# ARAG Claims Intelligence Platform

A comprehensive data science analysis of legal expense insurance claims, built with Streamlit.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy the data file to this directory
cp "../Data Science Sample data.csv" .

# Run the application
streamlit run Home.py
```

## Project Structure

```
arag_analysis/
├── Home.py                           # Executive dashboard & overview
├── pages/
│   ├── 1_Claims_Analytics.py         # EDA & claims patterns
│   ├── 2_Cost_Analysis.py            # Financial insights
│   ├── 3_Litigator_Performance.py    # Panel analysis
│   ├── 4_Predictive_Modeling.py      # ML models
│   └── 5_Business_Recommendations.py # Strategic insights
├── utils/
│   ├── __init__.py
│   └── data_loader.py                # Data loading utilities
├── requirements.txt
└── README.md
```

## Analysis Overview

### 1. Claims Analytics
- Distribution analysis across claim types, categories, and business classes
- Temporal patterns and seasonality
- Partner (agent) performance analysis
- Reinsurance comparison

### 2. Cost Analysis
- Cost distribution and concentration (Pareto analysis)
- Cost drivers identification
- High-value claims deep dive
- Year-over-year trends

### 3. Litigator Performance
- Panel vs Non-Panel comparison
- Statistical significance testing
- Category-specific performance
- Strategic recommendations

### 4. Predictive Modeling
- Claim outcome prediction (classification)
- Cost incurrence prediction (binary classification)
- Quick resolution prediction (classification)
- Customer segmentation (K-Means clustering)
- Feature importance analysis

### 5. Business Recommendations
- Operational efficiency improvements
- Cost management strategies
- Partner optimization
- ML implementation roadmap

## Key Insights

- **Total Claims Analyzed:** 100,000+
- **Total Costs:** £X million
- **ML Model Performance:** Accuracy, Precision, Recall metrics
- **Identified Savings Potential:** 8-10% cost reduction
