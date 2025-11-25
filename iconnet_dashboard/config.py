# -*- coding: utf-8 -*-
"""
Configuration settings for ICONNET Dashboard
"""
import streamlit as st
from pathlib import Path
import os

# Page Configuration
PAGE_CONFIG = {
    "page_title": "ICONNET Predictive Analytics Dashboard",
    "page_icon": "📊",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Paths
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "SourceData"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
REPORTS_DIR = ROOT_DIR / "reports"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR]:
    directory.mkdir(exist_ok=True)

# Dashboard sections
DASHBOARD_SECTIONS = [
    "Data Overview",
    "Predictive Modeling",
    "Churn Analysis",
    "Customer Segmentation",
    "Explainable AI",
    "Strategic Recommendations",
    "Model Management"
]

# Model parameters
MODEL_CONFIG = {
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": 42,
        "max_features": "sqrt"
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 7,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "scale_pos_weight": 1,  # Will be calculated based on class imbalance
        "gamma": 0.1
    },
    "kmeans": {
        "n_clusters": 5,
        "random_state": 42,
        "n_init": 10
    }
}

# Color schemes
COLOR_SCHEMES = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "danger": "#d62728",
    "warning": "#ffc107",
    "info": "#17a2b8",
    "light": "#f8f9fa",
    "dark": "#343a40"
}

# SHAP Configuration
SHAP_CONFIG = {
    "max_display": 15,
    "sample_size": 100,
    "plot_type": "bar",
    "color_scheme": "RdBu_r"
}

# Feature groups for analysis
FEATURE_GROUPS = {
    "demographic": ['tenure', 'contract_duration'],
    "behavioral": ['monthly_usage_gb', 'downtime_minutes', 'complaint_count'],
    "financial": ['monthly_charges', 'total_monthly_revenue', 'payment_delay_days'],
    "satisfaction": ['customer_satisfaction', 'churn']
}

# Business thresholds
BUSINESS_THRESHOLDS = {
    "high_value_revenue": 500000,  # Rp 500,000
    "high_churn_risk": 0.7,  # 70% probability
    "low_satisfaction": 6,  # Score <= 6
    "high_downtime": 300,  # Minutes per month
    "payment_delay_alert": 7  # Days
}

# CSS Styles
CUSTOM_CSS = """
<style>
    .main-header {font-size: 2.5rem; color: %s; font-weight: 700;}
    .sub-header {font-size: 1.8rem; color: %s; font-weight: 600;}
    .metric-label {font-size: 1.1rem; color: %s; font-weight: 500;}
    .highlight {background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0;}
    .footer {font-size: 0.9rem; color: #666; text-align: center; margin-top: 30px;}
    .team-info {background-color: #e8f4f8; padding: 15px; border-radius: 10px; margin: 10px 0;}
    .logo-container {display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;}
    .logo {height: 80px; object-fit: contain;}
    .header-title {text-align: center; flex-grow: 1; margin: 0 20px;}
    .info-box {background-color: #e3f2fd; padding: 15px; border-radius: 8px; border-left: 4px solid #2196f3; margin: 10px 0;}
    .warning-box {background-color: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107; margin: 10px 0;}
    .success-box {background-color: #d4edda; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745; margin: 10px 0;}
    .error-box {background-color: #f8d7da; padding: 15px; border-radius: 8px; border-left: 4px solid #dc3545; margin: 10px 0;}
    .shap-box {background-color: #fff3e0; padding: 15px; border-radius: 8px; border-left: 4px solid #ff9800; margin: 10px 0;}
    .stProgress .st-bo {background-color: %s;}
    
    /* Custom SHAP styling */
    .shap-feature-1 {color: #d62728; font-weight: bold;}
    .shap-feature-2 {color: #ff7f0e; font-weight: bold;}
    .shap-feature-3 {color: #2ca02c; font-weight: bold;}
    
    /* Paper-style formatting */
    .paper-section {background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 10px 0;}
    .paper-title {color: #1f77b4; border-bottom: 2px solid #1f77b4; padding-bottom: 10px;}
</style>
""" % (COLOR_SCHEMES["primary"], COLOR_SCHEMES["secondary"], COLOR_SCHEMES["success"], COLOR_SCHEMES["primary"])

# Export configuration
EXPORT_CONFIG = {
    "formats": ["csv", "xlsx", "json"],
    "include_timestamp": True,
    "default_format": "csv"
}