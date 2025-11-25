# -*- coding: utf-8 -*-
"""
Utility functions for ICONNET Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import functools
from datetime import datetime
import time

# Setup logging
def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('iconnet_dashboard.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def handle_error(func):
    """Decorator to handle errors in functions"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            display_error(f"Error in {func.__name__}: {str(e)}")
            return None
    return wrapper

def display_success(message: str):
    """Display success message"""
    st.success(f"✅ {message}")

def display_warning(message: str):
    """Display warning message"""
    st.warning(f"⚠️ {message}")

def display_info(message: str):
    """Display info message"""
    st.info(f"ℹ️ {message}")

def display_error(message: str):
    """Display error message"""
    st.error(f"❌ {message}")

def validate_dataframe(df: pd.DataFrame) -> bool:
    """Validate dataframe structure"""
    if df is None:
        display_error("DataFrame is None")
        return False
    
    if df.empty:
        display_error("DataFrame is empty")
        return False
    
    required_columns = ['customer_id', 'churn']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        display_warning(f"Missing columns: {missing_columns}")
        return False
    
    return True

def create_progress_tracker(steps: List[str]):
    """Create a progress tracker"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    return {'bar': progress_bar, 'text': status_text, 'steps': steps, 'current': 0}

def update_progress(tracker, message: str):
    """Update progress tracker"""
    tracker['current'] += 1
    progress = tracker['current'] / len(tracker['steps'])
    tracker['bar'].progress(progress)
    tracker['text'].text(f"{message} ({tracker['current']}/{len(tracker['steps'])})")

def complete_progress(tracker, final_message: str):
    """Complete progress tracker"""
    tracker['bar'].progress(1.0)
    tracker['text'].text(final_message)
    time.sleep(1)
    tracker['text'].empty()

def get_color_palette(n_colors: int) -> List[str]:
    """Get color palette for visualizations"""
    import plotly.express as px
    return px.colors.qualitative.Set1[:n_colors]

def format_currency(amount: float) -> str:
    """Format currency in Indonesian Rupiah"""
    if amount >= 1e9:
        return f"Rp {amount/1e9:.2f}B"
    elif amount >= 1e6:
        return f"Rp {amount/1e6:.2f}M"
    elif amount >= 1e3:
        return f"Rp {amount/1e3:.2f}K"
    else:
        return f"Rp {amount:,.0f}"

def format_percentage(value: float) -> str:
    """Format percentage"""
    return f"{value:.1f}%"

def calculate_business_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate key business metrics"""
    metrics = {}
    
    if 'total_monthly_revenue' in df.columns:
        metrics['total_revenue'] = df['total_monthly_revenue'].sum()
        metrics['avg_revenue'] = df['total_monthly_revenue'].mean()
        metrics['revenue_growth'] = 0  # Placeholder for actual growth calculation
    
    if 'churn' in df.columns:
        metrics['churn_rate'] = df['churn'].mean() * 100
        metrics['retention_rate'] = 100 - metrics['churn_rate']
        metrics['churned_customers'] = df['churn'].sum()
    
    if 'customer_satisfaction' in df.columns:
        metrics['avg_satisfaction'] = df['customer_satisfaction'].mean()
        metrics['satisfaction_rate'] = (metrics['avg_satisfaction'] / 10) * 100
    
    return metrics

def generate_sample_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate sample business insights"""
    insights = {}
    
    # Churn insights
    if 'churn' in df.columns:
        churn_corr = df.corr()['churn'].sort_values(ascending=False)
        insights['top_churn_factors'] = churn_corr[1:4].to_dict()  # Exclude churn itself
    
    # Revenue insights
    if 'total_monthly_revenue' in df.columns:
        high_value_threshold = df['total_monthly_revenue'].quantile(0.8)
        insights['high_value_customers'] = len(df[df['total_monthly_revenue'] > high_value_threshold])
        insights['high_value_percentage'] = (insights['high_value_customers'] / len(df)) * 100
    
    # Segmentation insights
    if 'segment' in df.columns:
        segment_performance = df.groupby('segment').agg({
            'total_monthly_revenue': 'mean',
            'churn': 'mean'
        }).round(3)
        insights['segment_performance'] = segment_performance.to_dict()
    
    return insights

@handle_error
def safe_column_operation(df: pd.DataFrame, column: str, operation: Callable) -> Any:
    """Safely perform operation on dataframe column"""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    return operation(df[column])

def check_missing_values(df: pd.DataFrame) -> Dict[str, int]:
    """Check for missing values in dataframe"""
    missing = df.isnull().sum()
    return missing[missing > 0].to_dict()

def get_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate data quality report"""
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': check_missing_values(df),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    return report