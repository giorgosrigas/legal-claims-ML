"""
Shared styling for ARAG Claims Intelligence Platform
=====================================================
Professional light/dark theme with consistent styling across all pages.
"""

import streamlit as st

# Color palette
COLORS = {
    'primary': '#1E3A5F',       # Deep navy blue
    'secondary': '#3498db',     # Light blue
    'accent': '#2ecc71',        # Green
    'warning': '#f39c12',       # Orange
    'danger': '#e74c3c',        # Red
    'bg_light': '#f8fafc',      # Very light gray-blue
    'bg_card': '#ffffff',       # White
    'bg_dark': '#1a1a2e',       # Dark blue-gray
    'text_primary': '#1E3A5F',  # Dark text
    'text_secondary': '#64748b', # Muted text
    'border': '#e2e8f0',        # Light border
}


def apply_global_styles():
    """Apply global CSS styles to the Streamlit app."""
    st.markdown("""
    <style>
        /* Global page styling */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        }

        /* Main content area */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        /* Headers styling */
        h1 {
            color: #1E3A5F !important;
            font-weight: 700 !important;
            border-bottom: 3px solid #3498db;
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem !important;
        }

        h2 {
            color: #1E3A5F !important;
            font-weight: 600 !important;
            margin-top: 2rem !important;
        }

        h3 {
            color: #2c5282 !important;
            font-weight: 600 !important;
        }

        h4 {
            color: #3182ce !important;
            font-weight: 500 !important;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1E3A5F 0%, #2c5282 100%);
        }

        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] .stMarkdown h1,
        [data-testid="stSidebar"] .stMarkdown h2,
        [data-testid="stSidebar"] .stMarkdown h3 {
            color: #ffffff !important;
        }

        /* Sidebar navigation links */
        [data-testid="stSidebar"] a {
            color: #e2e8f0 !important;
        }

        [data-testid="stSidebar"] a:hover {
            color: #ffffff !important;
        }

        /* Sidebar labels */
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stMultiSelect label,
        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stNumberInput label,
        [data-testid="stSidebar"] .stDateInput label {
            color: #ffffff !important;
            font-weight: 500 !important;
        }

        /* Sidebar input fields - white background for visibility */
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] .stSelectbox > div > div,
        [data-testid="stSidebar"] .stMultiSelect > div > div,
        [data-testid="stSidebar"] .stNumberInput > div > div > input {
            background-color: #ffffff !important;
            color: #1E3A5F !important;
            border-radius: 6px !important;
        }

        /* Sidebar selectbox text */
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background-color: #ffffff !important;
            color: #1E3A5F !important;
        }

        /* Sidebar number input buttons */
        [data-testid="stSidebar"] .stNumberInput button {
            background-color: #ffffff !important;
            color: #1E3A5F !important;
            border: 1px solid #e2e8f0 !important;
        }

        /* Sidebar page navigation */
        [data-testid="stSidebarNav"] a {
            color: #e2e8f0 !important;
            padding: 0.5rem 1rem !important;
            border-radius: 6px !important;
            margin: 2px 0 !important;
        }

        [data-testid="stSidebarNav"] a:hover {
            background-color: rgba(255, 255, 255, 0.1) !important;
            color: #ffffff !important;
        }

        [data-testid="stSidebarNav"] a[aria-selected="true"] {
            background-color: rgba(255, 255, 255, 0.2) !important;
            color: #ffffff !important;
            font-weight: 600 !important;
        }

        /* Sidebar nav span text */
        [data-testid="stSidebarNav"] span {
            color: #e2e8f0 !important;
        }

        [data-testid="stSidebarNav"] a[aria-selected="true"] span {
            color: #ffffff !important;
        }

        /* Metric cards */
        [data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            font-weight: 700 !important;
            color: #1E3A5F !important;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.9rem !important;
            color: #64748b !important;
            font-weight: 500 !important;
        }

        div[data-testid="metric-container"] {
            background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1rem 1.25rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05),
                        0 2px 4px -1px rgba(0, 0, 0, 0.03);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        div[data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08),
                        0 4px 6px -2px rgba(0, 0, 0, 0.04);
        }

        /* Info boxes */
        .stAlert {
            background: linear-gradient(145deg, #e3f2fd 0%, #bbdefb 100%);
            border: none;
            border-left: 4px solid #1E3A5F;
            border-radius: 0 8px 8px 0;
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #f8fafc;
            border-radius: 8px;
            font-weight: 600;
            color: #1E3A5F;
        }

        .streamlit-expanderContent {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 0 0 8px 8px;
        }

        /* DataFrame styling */
        .stDataFrame {
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }

        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #1E3A5F 0%, #2c5282 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            font-weight: 600;
            transition: all 0.2s ease;
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #2c5282 0%, #3182ce 100%);
            box-shadow: 0 4px 12px rgba(30, 58, 95, 0.3);
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #f1f5f9;
            border-radius: 10px;
            padding: 4px;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            border-radius: 8px;
            color: #64748b;
            font-weight: 500;
        }

        .stTabs [aria-selected="true"] {
            background-color: #ffffff;
            color: #1E3A5F;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }

        /* Divider styling */
        hr {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, #cbd5e1, transparent);
            margin: 2rem 0;
        }

        /* Plotly chart container */
        .js-plotly-plot {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
            background: transparent !important;
        }

        /* Plotly chart wrapper - remove white background */
        .stPlotlyChart {
            background: transparent !important;
        }

        /* Plotly chart inner container */
        .user-select-none {
            background: transparent !important;
        }

        /* Additional plotly elements */
        .plot-container, .svg-container {
            background: transparent !important;
        }

        /* Custom card styling */
        .custom-card {
            background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }

        .insight-card {
            background: linear-gradient(145deg, #f0f9ff 0%, #e0f2fe 100%);
            border-left: 4px solid #0ea5e9;
            border-radius: 0 12px 12px 0;
            padding: 1.25rem;
            margin: 1rem 0;
        }

        .warning-card {
            background: linear-gradient(145deg, #fffbeb 0%, #fef3c7 100%);
            border-left: 4px solid #f59e0b;
            border-radius: 0 12px 12px 0;
            padding: 1.25rem;
            margin: 1rem 0;
        }

        .success-card {
            background: linear-gradient(145deg, #f0fdf4 0%, #dcfce7 100%);
            border-left: 4px solid #22c55e;
            border-radius: 0 12px 12px 0;
            padding: 1.25rem;
            margin: 1rem 0;
        }

        .danger-card {
            background: linear-gradient(145deg, #fef2f2 0%, #fee2e2 100%);
            border-left: 4px solid #ef4444;
            border-radius: 0 12px 12px 0;
            padding: 1.25rem;
            margin: 1rem 0;
        }

        /* Section container */
        .section-container {
            background: #ffffff;
            border-radius: 16px;
            padding: 2rem;
            margin: 1.5rem 0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05),
                        0 2px 4px -1px rgba(0, 0, 0, 0.03);
        }

        /* Footer styling */
        .footer {
            text-align: center;
            padding: 2rem;
            color: #64748b;
            border-top: 1px solid #e2e8f0;
            margin-top: 3rem;
        }

        /* Spinner styling */
        .stSpinner > div {
            border-top-color: #1E3A5F !important;
        }
    </style>
    """, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = None):
    """Create a styled page header."""
    st.markdown(f"""
    <div style='text-align: center; padding: 1rem 0 2rem 0;'>
        <h1 style='font-size: 2.2rem; margin-bottom: 0.5rem; border: none;'>{title}</h1>
        {f"<p style='color: #64748b; font-size: 1.1rem;'>{subtitle}</p>" if subtitle else ""}
    </div>
    """, unsafe_allow_html=True)


def section_header(title: str, description: str = None):
    """Create a styled section header."""
    html = f"<h2 style='margin-top: 1rem;'>{title}</h2>"
    if description:
        html += f"<p style='color: #64748b; margin-bottom: 1.5rem;'>{description}</p>"
    st.markdown(html, unsafe_allow_html=True)


def info_card(title: str, content: str, card_type: str = "insight"):
    """Create a styled info card. Types: insight, warning, success, danger"""
    st.markdown(f"""
    <div class="{card_type}-card">
        <h4 style='margin-top: 0; margin-bottom: 0.5rem;'>{title}</h4>
        <p style='margin-bottom: 0;'>{content}</p>
    </div>
    """, unsafe_allow_html=True)


def style_plotly_chart(fig):
    """
    Apply consistent styling to Plotly charts to match app theme.
    Makes chart background transparent to blend with page background.
    """
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot area
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper/background
        font=dict(
            family="'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            color='#1E3A5F'
        ),
        title_font=dict(
            size=16,
            color='#1E3A5F',
            family="'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
        ),
        hoverlabel=dict(
            bgcolor='white',
            font_size=13,
            font_family="'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
        ),
        margin=dict(l=10, r=10, t=40, b=10)
    )

    # Style axes
    fig.update_xaxes(
        gridcolor='rgba(203, 213, 225, 0.3)',  # Light grid lines
        showgrid=True,
        zeroline=False,
        color='#64748b'
    )

    fig.update_yaxes(
        gridcolor='rgba(203, 213, 225, 0.3)',  # Light grid lines
        showgrid=True,
        zeroline=False,
        color='#64748b'
    )

    return fig
