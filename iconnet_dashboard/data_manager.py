# -*- coding: utf-8 -*-
"""
Data management module for ICONNET Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from config import DATA_DIR
from utils import handle_error, display_success, display_warning, display_error, logger

class DataManager:
    """Handles data loading, generation, and validation"""
    
    def __init__(self):
        self.data_dir = DATA_DIR 
        self.required_columns = [
            'customer_id', 'segment', 'tenure', 'contract_duration', 
            'monthly_charges', 'service_type',
            'additional_services', 'monthly_usage_gb', 'downtime_minutes',
            'customer_satisfaction', 'payment_method', 'complaint_count',
            'payment_delay_days', 'churn', 'total_monthly_revenue'
        ]
    
    @handle_error
    def get_available_files(self) -> list:
        """Get list of available CSV files"""
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
            return []
        
        csv_files = [f.name for f in self.data_dir.glob("*.csv")]
        return sorted(csv_files)
    
    @handle_error
    def save_uploaded_file(self, uploaded_file) -> bool:
        """Save uploaded file to data directory"""
        try:
            file_path = self.data_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            display_success(f"File {uploaded_file.name} berhasil diupload!")
            logger.info(f"File uploaded: {uploaded_file.name}")
            return True
        except Exception as e:
            display_error(f"Gagal menyimpan file: {str(e)}")
            return False
    
    @st.cache_data(ttl=1800)  # Cache for 30 minutes
    def load_csv_file(_self, file_path: str) -> Optional[pd.DataFrame]:
        """Load CSV file with error handling"""
        try:
            full_path = _self.data_dir / file_path
            if not full_path.exists():
                display_error(f"File tidak ditemukan: {file_path}")
                return None
            
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(full_path, encoding=encoding)
                    logger.info(f"File loaded successfully: {file_path} with encoding {encoding}")
                    return df
                except UnicodeDecodeError:
                    continue
            
            display_error(f"Tidak dapat membaca file dengan encoding yang tersedia: {file_path}")
            return None
            
        except Exception as e:
            display_error(f"Error loading file {file_path}: {str(e)}")
            return None
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate dataframe and return validation results"""
        if df is None or df.empty:
            return {"valid": False, "message": "DataFrame kosong atau None"}
        
        validation_results = {
            "valid": True,
            "message": "Data valid",
            "missing_columns": [],
            "data_types": {},
            "missing_values": {},
            "warnings": []
        }
        
        # Check required columns
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            validation_results["missing_columns"] = list(missing_cols)
            validation_results["warnings"].append(f"Missing columns: {missing_cols}")
        
        # Check data types
        for col in df.columns:
            validation_results["data_types"][col] = str(df[col].dtype)
        
        # Check missing values
        missing_values = df.isnull().sum()
        validation_results["missing_values"] = {
            col: int(count) for col, count in missing_values.items() if count > 0
        }
        
        # Additional validations
        if 'churn' in df.columns:
            unique_churn = df['churn'].unique()
            if not set(unique_churn).issubset({0, 1}):
                validation_results["warnings"].append("Churn column should contain only 0 and 1")
        
        if 'customer_satisfaction' in df.columns:
            satisfaction_range = df['customer_satisfaction'].dropna()
            if len(satisfaction_range) > 0:
                min_sat, max_sat = satisfaction_range.min(), satisfaction_range.max()
                if min_sat < 1 or max_sat > 10:
                    validation_results["warnings"].append("Customer satisfaction should be between 1-10")
        
        return validation_results
    
    def save_df_to_csv(_self, df: pd.DataFrame, file_path: Path) -> None:
        """Save dataframe to CSV file"""
        try:
            df.to_csv(file_path, index=False)
            logger.info(f"Data saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save data to CSV: {e}")
    
    @handle_error
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate and return a summary of the dataframe."""
        if df is None or df.empty:
            return {
                'total_customers': 0,
                'avg_revenue': 0,
                'avg_tenure': 0
            }
    
        summary = {
            'total_customers': len(df)
        }
    
        if 'total_monthly_revenue' in df.columns:
            summary['avg_revenue'] = df['total_monthly_revenue'].mean()
        if 'tenure' in df.columns:
            summary['avg_tenure'] = df['tenure'].mean()
    
        return summary