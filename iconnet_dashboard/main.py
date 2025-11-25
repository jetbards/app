# -*- coding: utf-8 -*-
"""
Main ICONNET Predictive Analytics Dashboard
Improved version with modular architecture and better code quality
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from contextlib import contextmanager, redirect_stdout
from io import StringIO
warnings.filterwarnings('ignore')

# Import custom modules
from config import PAGE_CONFIG, CUSTOM_CSS, DASHBOARD_SECTIONS, COLOR_SCHEMES
from utils import (
    setup_logging, display_success, display_warning, display_info, display_error,
    validate_dataframe, create_progress_tracker, update_progress, complete_progress
)
from data_manager import DataManager
from models import ChurnPredictionModel, CustomerSegmentationModel, ExplainableAI, LIME_AVAILABLE, SHAP_AVAILABLE
from visualizations import DashboardVisualizations

# Initialize logging
logger = setup_logging()

# Set page configuration
st.set_page_config(**PAGE_CONFIG)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

class IconnetDashboard:
    """Main dashboard class"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.viz = DashboardVisualizations()
        self.churn_model = ChurnPredictionModel()
        self.segmentation_model = CustomerSegmentationModel()
        
    def display_header(self):
        """Display header with logos and title"""
        try:
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                # PLN ICONNET Logo placeholder
                if os.path.exists("LogoPLNIconnet.jpg"):
                    st.image("LogoPLNIconnet.jpg", width=100)
                else:
                    st.markdown("""
                    <div style="text-align: center; background-color: #1a5276; color: white; padding: 20px; border-radius: 10px;">
                        <h3 style="margin: 0;">PLN ICONNET</h3>
                        <p style="margin: 0; font-size: 0.8rem;">SEMUA MAKIN MUDAH</p>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <div class="header-title">
                    <h2 style="color: #1f77b4; margin: 0; text-align: center;">PEMANFAATAN ANALISIS PREDIKTIF</h2>
                    <h4 style="color: #666; margin: 0; text-align: center;">OPTIMALISASI PENDAPATAN LAYANAN ICONNET</h4>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                # BINUS Logo placeholder
                if os.path.exists("LogoBinus.png"):
                    st.image("LogoBinus.png", width=100)
                else:
                    st.markdown("""
                    <div style="text-align: center; background-color: #e74c3c; color: white; padding: 20px; border-radius: 10px;">
                        <h3 style="margin: 0;">BINUS</h3>
                        <p style="margin: 0; font-size: 0.8rem;">UNIVERSITY</p>
                    </div>
                    """, unsafe_allow_html=True)
                
        except Exception as e:
            logger.error(f"Error displaying header: {e}")
            st.markdown("## ICONNET Predictive Analytics Dashboard")

    def display_team_info(self):
        """Display team information"""
        st.markdown("""
        <div class="team-info" style="background-color: #f8f9fa; padding: 8px; border-radius: 5px; margin: 5px 0;">
			<p style="text-align: center; margin: 2px 0; font-size: 1.2rem; font-weight: bold;">PT PLN ICON PLUS SBU REGIONAL JAWA TENGAH</p>
			<p style="text-align: center; color: #1f77b4; margin: 1px 0; font-size: 0.75rem;">Dibuat Oleh:</p>
			<div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 2px;">
				<div style="text-align: center;">
					<p style="font-weight: bold; margin: 0; color: #2c3e50; font-size: 0.75rem;">Hani Setiawan</p>
					<p style="margin: 0; color: #7f8c8d; font-size: 0.65rem;">2702464202</p>
				</div>
				<div style="text-align: center;">
					<p style="font-weight: bold; margin: 0; color: #2c3e50; font-size: 0.75rem;">Jetbar R. H. D.</p>
					<p style="margin: 0; color: #7f8c8d; font-size: 0.65rem;">2702462973</p>
				</div>
				<div style="text-align: center;">
					<p style="font-weight: bold; margin: 0; color: #2c3e50; font-size: 0.75rem;">Naufal Yafi</p>
					<p style="margin: 0; color: #7f8c8d; font-size: 0.65rem;">2702476240</p>
				</div>
			</div>
		</div>
        """, unsafe_allow_html=True)

    def load_data_section(self):
        """Handle data loading and validation"""
        st.sidebar.subheader("📂 Data Management")
        
        # Data source selection
        data_source = st.sidebar.radio("Pilih Sumber Data:", ["Upload File CSV", "Gunakan File Tersimpan"])
        df = None
        if data_source == "Upload File CSV":
            uploaded_file = st.sidebar.file_uploader("Choose CSV file", type=['csv'])
            
            if uploaded_file is not None:
                if self.data_manager.save_uploaded_file(uploaded_file):
                    df = self.data_manager.load_csv_file(uploaded_file.name)
                    if df is not None:
                        st.session_state['data'] = df
        elif data_source == "Gunakan File Tersimpan":
            available_files = self.data_manager.get_available_files()
            
            if available_files:
                selected_file = st.sidebar.selectbox("Select file:", available_files)
                
                if st.sidebar.button("Load Data"):
                    df = self.data_manager.load_csv_file(selected_file)
                    if df is not None:
                        st.session_state['data'] = df
            else:
                display_info("Tidak ada file CSV. Silakan upload file terlebih dahulu.")
        
        # Return data from session state if available
        return st.session_state.get('data', None)

    def data_overview_section(self, df):
        """Display data overview section"""
        st.header("📋 Data Overview")
        
        # Validate data
        validation_results = self.data_manager.validate_data(df)
        
        if validation_results.get("warnings"):
            for warning in validation_results["warnings"]:
                display_warning(warning)
        
        # Data summary
        col1, col2 = st.columns([2, 1])
        
        # Buat salinan DataFrame untuk tampilan dan hapus kolom 'churn'
        display_df = df.copy()
        if 'churn' in display_df.columns:
            display_df = display_df.drop(columns=['churn'])
            
        # Konversi kolom datetime ke string untuk mencegah error serialisasi Arrow
        for col in display_df.select_dtypes(include=['datetime64[ns]']).columns:
            # Format tanggal ke YYYY-MM-DD, dan biarkan NaT (Not a Time) jika ada
            display_df[col] = display_df[col].dt.strftime('%Y-%m-%d').replace({pd.NaT: None})

        # --- NEW: Explicitly set customer_id as object for display purposes ---
        if 'customer_id' in display_df.columns:
            display_df['customer_id'] = display_df['customer_id'].astype(str)


        with col1:
            st.subheader("Sample Data")
            st.dataframe(display_df.head(10), use_container_width=True)

            # Tombol untuk menyimpan data sampel
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Simpan Sample Data (CSV)",
                data=csv_data,
                file_name="iconnet_sample_data.csv",
                mime="text/csv",
                help="Klik untuk mengunduh data yang sedang ditampilkan sebagai file CSV"
            )
            
            # Data info
            with st.expander("📊 Data Information"):
                st.metric("Shape", f"{display_df.shape[0]} rows × {display_df.shape[1]} columns")

                # --- REFACTORED: Use tabs for cleaner separation of stats ---
                num_tab, cat_tab = st.tabs(["Numerical Summary", "Categorical Summary"])

                with num_tab:
                    # Select only numeric columns for description
                    numeric_df = display_df.select_dtypes(include=np.number)
                    st.markdown(f"##### Ringkasan Statistik untuk Fitur Numerik ({numeric_df.shape[1]} columns)")
                    if not numeric_df.empty:
                        # Calculate summary
                        num_summary = numeric_df.describe().transpose()
                        num_summary['missing'] = numeric_df.isnull().sum()
                        num_summary = num_summary[['count', 'missing', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
                        
                        # Format for display
                        st.dataframe(num_summary.style.format({
                            'count': '{:,.0f}',
                            'missing': '{:,.0f}',
                            'mean': '{:,.2f}',
                            'std': '{:,.2f}',
                            'min': '{:,.2f}',
                            '25%': '{:,.2f}',
                            '50%': '{:,.2f}',
                            '75%': '{:,.2f}',
                            'max': '{:,.2f}'
                        }, na_rep="-"), use_container_width=True)
                    else:
                        st.info("Tidak ada fitur numerik yang ditemukan dalam data.")

                with cat_tab:
                    # Select non-numeric columns
                    categorical_df = display_df.select_dtypes(include=['object', 'category'])
                    st.markdown(f"##### Ringkasan Statistik untuk Fitur Kategorikal ({categorical_df.shape[1]} columns)")
                    if not categorical_df.empty:
                        # Calculate summary
                        cat_summary = categorical_df.describe().transpose()
                        cat_summary['missing'] = categorical_df.isnull().sum()
                        cat_summary['missing (%)'] = (cat_summary['missing'] / len(display_df)) * 100
                        cat_summary = cat_summary[['count', 'missing', 'missing (%)']]
                        
                        # Format for display
                        st.dataframe(cat_summary.style.format({
                            'count': '{:,.0f}',
                            'missing': '{:,.0f}',
                            'missing (%)': '{:.2f}%'
                        }, na_rep="-"), use_container_width=True)
                    else:
                        st.info("Tidak ada fitur kategorikal yang ditemukan dalam data.")
        
        with col2:
            st.subheader("Key Metrics")
            summary = self.data_manager.get_data_summary(df)
            
            # Display metrics
            # --- REFACTORED: Added churn rate to key metrics if available ---
            metrics = {
                "Total Customers": summary.get('total_customers', 0),
                "Average Revenue": {
                    'value': summary.get('avg_revenue', 0),
                    'format': 'currency'
                },
                "Average Tenure": f"{summary.get('avg_tenure', 0):.1f} months",
            }
            
            # Conditionally add churn rate if the column exists
            if 'churn' in df.columns:
                metrics["Churn Rate"] = {
                    'value': df['churn'].mean() * 100,
                    'format': 'percentage'
                }
            
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, dict):
                    format_type = metric_value.get('format', 'number')
                    value = metric_value.get('value', 0)
                    
                    if format_type == 'currency':
                        st.metric(metric_name, f"Rp {value:,.0f}")
                    elif format_type == 'percentage':
                        st.metric(metric_name, f"{value:.1f}%")
                    else:
                        st.metric(metric_name, f"{value:,.0f}")
                else:
                    st.metric(metric_name, metric_value)
        
        # Visualizations
        st.subheader("📊 Data Distribution")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'total_monthly_revenue' in df.columns:
                fig_revenue = self.viz.create_revenue_distribution(display_df)
                st.plotly_chart(fig_revenue, use_container_width=True)
        
        with col2:
            if 'segment' in df.columns:
                fig_segment = self.viz.create_segment_pie_chart(display_df)
                st.plotly_chart(fig_segment, use_container_width=True)
        
        with col3:
            if 'contract_duration' in df.columns:
                fig_contract = self.viz.create_contract_bar_chart(display_df)
                st.plotly_chart(fig_contract, use_container_width=True)

    def churn_analysis_section(self, df):
        """Display churn analysis section"""
        st.header("📉 Churn Analysis")
        
        # --- REFACTORED: Centralized check and data preparation ---
        # If models haven't been trained, display a message and exit.
        if 'model_results' not in st.session_state:
            display_warning("Latih model terlebih dahulu di bagian 'Predictive Modeling' untuk melihat analisis churn berdasarkan hasil prediksi.")
            return
        
        # Get model results and test data once
        results = st.session_state['model_results']
        test_data = results.get('test_data', {})
        X_test = test_data.get('X_test')

        if X_test is None or X_test.empty:
            display_error("Data tes tidak ditemukan dalam hasil model. Silakan latih ulang model.")
            return

        # Get predictions from the best available model
        if 'random_forest' in results:
            predictions = results['random_forest']['predictions']
        elif 'xgboost' in results:
            predictions = results['xgboost']['predictions']
        else:
            display_error("Tidak ada hasil prediksi yang ditemukan.")
            return

        # Create the primary analysis DataFrame from the test set with model predictions
        analysis_df = df.loc[X_test.index].copy()
        analysis_df['churn'] = predictions

        st.info("Menampilkan analisis berdasarkan pelanggan yang **diprediksi** akan churn oleh model pada data tes.")
        
        # =============================
        # 📊 Key Metrics
        # =============================
        # Key metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            total_churn = analysis_df['churn'].sum()
            st.metric("Total Churned", f"{total_churn:,}")
        
        with col2:
            if 'total_monthly_revenue' in df.columns:
                lost_revenue = analysis_df[analysis_df['churn'] == 1]['total_monthly_revenue'].sum()
                st.metric("Lost Revenue", f"Rp {lost_revenue:,.0f}")
        
        with col3:
            avg_tenure_churned = analysis_df[analysis_df['churn'] == 1]['tenure'].mean()
            st.metric("Avg Tenure (Churned)", f"{avg_tenure_churned:.1f} months")
        
        # =============================
        # 📈 Visual Analysis
        # =============================
        # Churn analysis charts
        st.subheader("🔍 Churn Analysis by Segments")
        churn_charts = self.viz.create_churn_analysis_charts(analysis_df) # Corrected: Use analysis_df only
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(churn_charts['churn_by_segment'], use_container_width=True)
            
            # Display segment data
            segment_data = analysis_df.groupby('segment')['churn'].agg(['count', 'sum']).reset_index()
            segment_data.columns = ['Segment', 'Total Customers', 'Churned']
            st.dataframe(segment_data.astype(str), use_container_width=True)
        
        with col2:
            st.plotly_chart(churn_charts['churn_by_service'], use_container_width=True)
            
            # Display service data
            service_data = analysis_df.groupby('service_type')['churn'].agg(['count', 'sum']).reset_index()
            service_data.columns = ['Service Type', 'Total Customers', 'Churned']
            st.dataframe(service_data.astype(str), use_container_width=True)
        
        # Churn factors analysis
        st.subheader("📊 Churn Factors Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(churn_charts['tenure_vs_churn'], use_container_width=True)
            st.plotly_chart(churn_charts['downtime_vs_churn'], use_container_width=True)
        
        with col2:
            st.plotly_chart(churn_charts['satisfaction_vs_churn'], use_container_width=True)            
            st.plotly_chart(churn_charts['payment_delay_vs_churn'], use_container_width=True)

        # Correlation heatmap of numerical features
        st.subheader("Heatmap Korelasi Fitur Numerik")
        
        with st.expander("ℹ️ Cara Membaca Heatmap Korelasi", expanded=False):
            st.markdown("""
            Heatmap ini adalah alat visual untuk memahami hubungan antara berbagai fitur numerik. Setiap kotak menunjukkan seberapa kuat hubungan antara dua variabel.

            - **Sumbu X dan Y**: Menampilkan fitur-fitur numerik yang dianalisis.
            - **Warna Kotak**:
                - **Biru Tua (mendekati +1)**: Korelasi Positif Kuat. Jika satu variabel naik, variabel lain cenderung ikut naik.
                - **Merah Tua (mendekati -1)**: Korelasi Negatif Kuat. Jika satu variabel naik, variabel lain cenderung turun.
                - **Warna Terang (mendekati 0)**: Korelasi Lemah atau tidak ada hubungan.
            - **Angka di Dalam Kotak**: Nilai koefisien korelasi yang presisi.

            **Fokus Utama untuk Analisis Churn:**
            Lihat baris atau kolom `churn`. Fitur dengan warna **biru tua** adalah pendorong churn (misal: `complaint_count`), sedangkan fitur dengan warna **merah tua** adalah pencegah churn (misal: `tenure` atau `customer_satisfaction`).
            """)

        st.info("Heatmap di bawah ini dibuat berdasarkan data pelanggan yang **diprediksi** akan churn dan tidak churn oleh model.")
        
        # Pilih fitur numerik yang relevan untuk korelasi
        corr_features = ['tenure', 'monthly_charges', 'monthly_usage_gb', 'downtime_minutes', 'customer_satisfaction', 'complaint_count', 'payment_delay_days', 'total_monthly_revenue', 'churn']
        available_corr_features = [f for f in corr_features if f in analysis_df.columns and pd.api.types.is_numeric_dtype(analysis_df[f])]

        if len(available_corr_features) > 1:
            corr_fig = self.viz.create_correlation_heatmap(analysis_df[available_corr_features], title="Korelasi antar Variabel Kunci")
            st.plotly_chart(corr_fig, use_container_width=True)
		
		        # =============================
        # 📋 Detail Customer Churn
        # =============================
        st.subheader("📋 Detail Pelanggan yang Diprediksi Churn")

        # --- REFACTORED: Use the already prepared analysis_df ---
        churn_customers_df = analysis_df[analysis_df['churn'] == 1]

        if churn_customers_df.empty:
            st.success("✅ Tidak ada pelanggan yang teridentifikasi berisiko churn saat ini.")
        else:
            st.write(f"Menampilkan {len(churn_customers_df)} pelanggan yang teridentifikasi berisiko churn:")

            # Pilih kolom penting untuk ditampilkan
            display_columns = [
                'customer_id', 'tanggalaktivasi', 'segment', 'service_type', 
                'tenure', 'customer_satisfaction', 
                'total_monthly_revenue', 'payment_method', 'complaint_count'
            ]
            available_columns = [col for col in display_columns if col in churn_customers_df.columns]

            # Tambahkan fitur pencarian
            search_term = st.text_input("🔍 Cari Customer (ID / Segment / Service Type):").lower()
            if search_term:
                filtered_df = churn_customers_df[
                    churn_customers_df[available_columns].astype(str).apply(lambda x: x.str.lower().str.contains(search_term)).any(axis=1)
                ]
            else:
                filtered_df = churn_customers_df


            st.dataframe(filtered_df[available_columns], use_container_width=True, hide_index=True)

            # Opsi ekspor data churned
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Data Pelanggan Berisiko Churn (CSV)",
                data=csv_data,
                file_name="customer_risk_churn_detail.csv",
                mime="text/csv"
            )


    def predictive_modeling_section(self, df):
        """Display predictive modeling section"""
        st.header("🔮 Predictive Modeling")
        st.subheader("Churn Prediction using Machine Learning")
        
        if 'churn' not in df.columns:
            display_warning("Churn column not found. Cannot perform churn prediction.")
            return
        
        with st.expander("ℹ️ Model Information", expanded=False):
            st.markdown("""
            **Models Used:**
            - **Random Forest**: Ensemble method with multiple decision trees for robust predictions
            - **XGBoost**: Gradient boosting with advanced optimization (if available)
            - **Class Balancing**: Cluster-based undersampling for handling imbalanced data
            
            **Features Used:**
            - Customer demographics and behavior
            - Service usage patterns
            - Payment and satisfaction metrics
            """)
        
        # Model training button
        if st.button("🚀 Train Models", type="primary"):
            
            progress_tracker = create_progress_tracker([
                "Preparing data",
                "Handling class imbalance", 
                "Training Random Forest",
                "Training XGBoost",
                "Evaluating models"
            ])
            
            try:
                # Prepare data
                update_progress(progress_tracker, "Preparing data...")
                
                # Create a copy of the dataframe for modeling to avoid side effects
                modeling_df = df.copy()
                if 'customer_id' in modeling_df.columns:
                    modeling_df = modeling_df.drop(columns=['customer_id'])
                
                X, y = self.churn_model.prepare_data(modeling_df)
                
                if X is None or y is None:
                    display_error("Failed to prepare data for modeling")
                    return
                
                # Train models
                update_progress(progress_tracker, "Training models...")
                results = self.churn_model.train_models(X, y)
                
                # Handle the case where the test set is not representative
                if results.get("error"):
                    display_error(results["error"])
                    # Stop further execution for this run
                    return

                if results:
                    complete_progress(progress_tracker, "Model training completed successfully!")
                    
                    # Store results in session state
                    st.session_state['model_results'] = results
                    st.session_state['churn_model'] = self.churn_model
                    
                    # Display results
                    self.display_model_results(results)
                    
            except Exception as e:
                display_error(f"Error in model training: {str(e)}")
                logger.error(f"Model training error: {e}")
        
        # Display existing results if available
        elif 'model_results' in st.session_state:
            self.display_model_results(st.session_state['model_results'])

    def display_model_results(self, results):
        """Display model training results"""
        st.subheader("🎯 Model Performance")
        
        # Model comparison
        col1, col2 = st.columns(2)
        
        # Random Forest Results
        with col1:
            if 'random_forest' in results:
                st.markdown("#### 🌲 Random Forest")
                rf_results = results['random_forest']
                
                # Safely get metrics for class '1' (Churn), default to 0 if not present
                report = rf_results.get('classification_report', {})
                churn_metrics = report.get('1', {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0})
                if '1' not in report:
                    display_warning("Random Forest model did not predict any churn instances. Churn metrics are 0.")

                # Metrics
                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric("Accuracy", f"{report.get('accuracy', 0.0):.3f}")
                    st.metric("Precision (Churn)", f"{churn_metrics['precision']:.3f}")
                with metrics_col2:
                    st.metric("Recall (Churn)", f"{churn_metrics['recall']:.3f}")
                    st.metric("F1-Score (Churn)", f"{churn_metrics['f1-score']:.3f}")
                
                # Confusion Matrix
                cm_fig = self.viz.create_confusion_matrix(rf_results['confusion_matrix'], "Random Forest Confusion Matrix")
                st.plotly_chart(cm_fig, use_container_width=True)
        
        # XGBoost Results
        with col2:
            if 'xgboost' in results:
                st.markdown("#### ⚡ XGBoost")
                xgb_results = results['xgboost']

                # Safely get metrics for class '1' (Churn), default to 0 if not present
                report = xgb_results.get('classification_report', {})
                churn_metrics = report.get('1', {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0})
                if '1' not in report:
                    display_warning("XGBoost model did not predict any churn instances. Churn metrics are 0.")

                # Metrics
                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric("Accuracy", f"{report.get('accuracy', 0.0):.3f}")
                    st.metric("Precision (Churn)", f"{churn_metrics['precision']:.3f}")
                with metrics_col2:
                    st.metric("Recall (Churn)", f"{churn_metrics['recall']:.3f}")
                    st.metric("F1-Score (Churn)", f"{churn_metrics['f1-score']:.3f}")
                
                # Confusion Matrix
                cm_fig = self.viz.create_confusion_matrix(xgb_results['confusion_matrix'], "XGBoost Confusion Matrix")
                st.plotly_chart(cm_fig, use_container_width=True)
        
        st.subheader("📈 Model Evaluation Curves")
        col1, col2 = st.columns(2)
        
        test_data = results.get('test_data', {})
        y_test = test_data.get('y_test')
        
        if y_test is not None:
            with col1:
                if 'random_forest' in results:
                    roc_fig = self.viz.create_roc_curve(
                        y_test,
                        results['random_forest']['probabilities'], 
                        "Random Forest",
                        roc_auc_value=results['random_forest'].get('roc_auc', 0.0)
                    )
                    st.plotly_chart(roc_fig, use_container_width=True)
            
            with col2:
                if 'xgboost' in results:
                    roc_fig = self.viz.create_roc_curve(
                        y_test, 
                        results['xgboost']['probabilities'], # Corrected from previous turn
                        "XGBoost",
                        roc_auc_value=results['xgboost'].get('roc_auc', 0.0)
                    )
                    st.plotly_chart(roc_fig, use_container_width=True)
        
        # Model Comparison 
        st.subheader("📊 Model Comparison")

        model_comparison_data = [] # Inisialisasi list untuk data perbandingan
        if 'random_forest' in results:
            rf_results = results['random_forest']
            rf_report = rf_results.get('classification_report', {})
            rf_churn_metrics = rf_report.get('1', {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0})
            model_comparison_data.append({
                'Model': 'Random Forest',
                'Precision': rf_churn_metrics['precision'],
                'Recall': rf_churn_metrics['recall'],
                'F1-Score': rf_churn_metrics['f1-score'],
                'ROC-AUC': rf_results.get('roc_auc', 0.0)
            })

        if 'xgboost' in results:
            xgb_results = results['xgboost']
            xgb_report = xgb_results.get('classification_report', {})
            xgb_churn_metrics = xgb_report.get('1', {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0})
            model_comparison_data.append({
                'Model': 'XGBoost',
                'Precision': xgb_churn_metrics['precision'],
                'Recall': xgb_churn_metrics['recall'],
                'F1-Score': xgb_churn_metrics['f1-score'],
                'ROC-AUC': xgb_results.get('roc_auc', 0.0)
            })

        if model_comparison_data:
            comparison_df = pd.DataFrame(model_comparison_data).set_index('Model')
            
            # 1. Table Comparison
            st.markdown("#### Perbandingan Metrik Model")
            st.dataframe(comparison_df.style.format("{:.4f}").highlight_max(axis=0, color=COLOR_SCHEMES['success']), use_container_width=True)

            # Penjelasan Metrik
            with st.expander("ℹ️ Penjelasan Metrik Evaluasi"):
                st.markdown("""
                - **Precision (Presisi)**: Dari semua pelanggan yang diprediksi akan *churn*, berapa persen yang benar-benar *churn*?
                  - *Fokus*: Meminimalkan prediksi *churn* yang salah (False Positive).
                  - *Contoh*: Presisi tinggi berarti model sangat akurat ketika memprediksi seorang pelanggan akan *churn*.
                - **Recall (Sensitivitas)**: Dari semua pelanggan yang sebenarnya *churn*, berapa persen yang berhasil diidentifikasi oleh model?
                  - *Fokus*: Menemukan sebanyak mungkin pelanggan yang benar-benar *churn* (meminimalkan False Negative).
                  - *Contoh*: Recall tinggi berarti model mampu menangkap sebagian besar pelanggan yang akan *churn*.
                - **F1-Score**: Rata-rata harmonik dari Precision dan Recall. Memberikan keseimbangan antara keduanya.
                  - *Fokus*: Berguna ketika biaya untuk False Positive dan False Negative sama-sama penting.
                - **ROC-AUC**: Kemampuan model untuk membedakan antara kelas positif (*churn*) dan negatif (*tidak churn*).
                  - *Skor*: 1.0 adalah sempurna, 0.5 sama dengan tebakan acak. Semakin tinggi skornya, semakin baik model dalam memisahkan pelanggan *churn* dan non-*churn*.
                """)

            # 2. Visual Comparison (barchart)
            st.markdown("#### Perbandingan Visual Model")
            
            comparison_melted = comparison_df.reset_index().melt(id_vars='Model', var_name='Metric', value_name='Score')
            fig = px.bar(comparison_melted, x='Metric', y='Score', color='Model', barmode='group', title='Perbandingan Kinerja Model', text_auto='.3f')
            fig.update_layout(yaxis_title="Score", xaxis_title="Metric")
            st.plotly_chart(fig, use_container_width=True)

        # Feature Importance
        st.subheader("🎯 Feature Importance")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'random_forest' in results:
                rf_importance_fig = self.viz.create_feature_importance_chart(
                    results['random_forest']['feature_importance'],
                    "Random Forest Feature Importance"
                )
                st.plotly_chart(rf_importance_fig, use_container_width=True)
        
        with col2:
            if 'xgboost' in results:
                xgb_importance_fig = self.viz.create_feature_importance_chart(
                    results['xgboost']['feature_importance'],
                    "XGBoost Feature Importance"
                )
                st.plotly_chart(xgb_importance_fig, use_container_width=True)
        
        # Interactive Feature Importance Comparison
        st.subheader("Interactive Feature Importance Comparison")

        if 'random_forest' in results and 'xgboost' in results:
            rf_importance = results['random_forest']['feature_importance']
            xgb_importance = results['xgboost']['feature_importance']

            # Create a DataFrame for combined feature importance
            importance_df = pd.DataFrame({
                'Feature': list(rf_importance.keys()),
                'Random Forest': list(rf_importance.values()),
                'XGBoost': list(xgb_importance.values())
            }).set_index('Feature')

            # Create a selection for features to compare
            selected_features = st.multiselect(
                "Pilih fitur untuk dibandingkan:",
                options=importance_df.index.tolist(),
                default=importance_df.index[:10].tolist()  # Show top 10 by default
            )

            if selected_features:
                # Filter the DataFrame based on selected features
                filtered_df = importance_df.loc[selected_features]

                # Create a bar chart using Plotly Express
                fig = px.bar(
                    filtered_df,
                    x=filtered_df.index,
                    y=['Random Forest', 'XGBoost'],
                    barmode='group',
                    title='Perbandingan Feature Importance',
                    labels={'value': 'Importance', 'index': 'Feature'},
                    color_discrete_sequence=px.colors.qualitative.Set1
                )

                fig.update_layout(xaxis_title="Fitur", yaxis_title="Importance")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Pilih setidaknya satu fitur untuk dibandingkan.")
        else:
            st.info("Latih kedua model (Random Forest dan XGBoost) untuk mengaktifkan perbandingan fitur.")

        # Save models button
        if st.button("💾 Save Models"):
            self.churn_model.save_models(results)

    def customer_segmentation_section(self, df):
        """Display customer segmentation section"""
        st.header("👥 Customer Segmentation")
        st.subheader("K-Means Clustering Analysis")
        
        # Segmentation parameters
        col1, col2 = st.columns([1, 3])
        
        with col1:
            n_clusters = st.slider("Jumlah Klaster (Segments):", 2, 5, 3, help="Pilih jumlah segmen pelanggan yang ingin diidentifikasi. Default adalah 3 untuk mencocokkan profil 'Loyal', 'At-Risk', dan 'New'.")
            
            if st.button("🔍 Perform Segmentation", type="primary"):
                with st.spinner("Melakukan segmentasi pelanggan..."):
                    # Create a context manager to suppress unwanted print statements
                    @contextmanager
                    def st_capture_stdout():
                        with StringIO() as stdout, redirect_stdout(stdout):
                            yield

                    with st_capture_stdout():
                        try:
                            results = self.segmentation_model.perform_segmentation(df, n_clusters)
                            
                            if results:
                                st.session_state['segmentation_results'] = results
                                display_success(f"Segmentation completed with silhouette score: {results['silhouette_score']:.3f}")
                            
                        except Exception as e:
                            display_error(f"Error in segmentation: {str(e)}")
                            logger.error(f"Segmentation error: {e}")
        
        with col2:
            # Display existing results if available
            if 'segmentation_results' in st.session_state:
                results = st.session_state['segmentation_results']
                
                st.metric("Skor Silhouette", f"{results['silhouette_score']:.3f}")
                st.info("Skor Silhouette berkisar dari -1 hingga 1. Skor yang lebih tinggi menunjukkan klaster yang lebih padat dan terdefinisi dengan baik.")
        
        # Display segmentation results
        if 'segmentation_results' in st.session_state:
            results = st.session_state['segmentation_results']
            clustered_df = results['clustered_data']
            
            # Cluster visualizations
            st.subheader("📊 Analisis Klaster")
            
            cluster_charts = self.viz.create_cluster_visualization(clustered_df)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(cluster_charts['cluster_distribution'], use_container_width=True)
                st.plotly_chart(cluster_charts['revenue_by_cluster'], use_container_width=True)
            
            with col2:
                st.plotly_chart(cluster_charts['revenue_vs_satisfaction'], use_container_width=True)
                if 'churn_by_cluster' in cluster_charts:
                    st.plotly_chart(cluster_charts['churn_by_cluster'], use_container_width=True)
            
            # Cluster summary table
            st.subheader("Ringkasan Karakteristik Klaster")
            cluster_summary = results['cluster_summary']
            st.dataframe(cluster_summary, use_container_width=True)
            
            # =================================================
            # SECTION: INTERPRETASI PROFIL PELANGGAN (BARU)
            # =================================================
            st.subheader("👤 Profil Pelanggan dan Rekomendasi Strategis")

            if n_clusters == 3:
                try:
                    # Logika untuk memetakan klaster ke profil
                    # Diasumsikan 'tenure' dan 'churn' adalah kolom MultiIndex
                    tenure_col = ('tenure', 'mean') if isinstance(cluster_summary.columns, pd.MultiIndex) else 'tenure'
                    churn_col = ('churn', 'mean') if isinstance(cluster_summary.columns, pd.MultiIndex) else 'churn'

                    # Urutkan berdasarkan tenure untuk identifikasi
                    sorted_clusters = cluster_summary.sort_values(by=tenure_col, ascending=False)

                    # Pemetaan berdasarkan urutan tenure
                    profile_mapping = {
                        sorted_clusters.index[0]: "Loyal Customer",
                        sorted_clusters.index[2]: "New Customer",
                        sorted_clusters.index[1]: "At-Risk Customer"
                    }

                    # Tampilkan profil dalam kolom
                    cols = st.columns(3)
                    
                    # Profil A: Loyal Customer
                    with cols[0]:
                        st.markdown("##### Profil A: Pelanggan Loyal")
                        st.markdown("- **Rata-rata Tenure**: Tinggi ( > 38 bulan)")
                        st.markdown("- **Tingkat Churn**: Sangat Rendah")
                        st.markdown("- **Rekomendasi**: Fokus pada program loyalitas, penawaran *upselling* (misal: upgrade ke paket lebih tinggi), dan jadikan mereka sebagai advokat merek.")

                    # Profil B: At-Risk Customer
                    with cols[1]:
                        st.markdown("##### Profil B: Pelanggan Berisiko (At-Risk)")
                        st.markdown("- **Rata-rata Tenure**: Sedang")
                        st.markdown("- **Tingkat Churn**: **Tinggi**")
                        st.markdown("- **Rekomendasi**: Lakukan intervensi proaktif, selesaikan keluhan dengan cepat, tawarkan kompensasi jika ada masalah layanan, dan berikan diskon retensi.")

                    # Profil C: New Customer
                    with cols[2]:
                        st.markdown("##### Profil C: Pelanggan Baru")
                        st.markdown("- **Rata-rata Tenure**: Rendah ( < 6 bulan)")
                        st.markdown("- **Tingkat Churn**: Sedang")
                        st.markdown("- **Rekomendasi**: Pastikan proses *onboarding* berjalan lancar, berikan panduan penggunaan layanan, dan lakukan pengecekan kepuasan di bulan-bulan awal.")

                except Exception as e:
                    st.warning(f"Gagal membuat profil otomatis: {e}")
            else:
                st.info("Profil pelanggan otomatis (Loyal, At-Risk, New) hanya ditampilkan ketika jumlah klaster diatur ke 3.")


            # Detailed cluster analysis
            with st.expander("🔍 Analisis Detail per Klaster"):
                selected_cluster = st.selectbox("Pilih klaster untuk analisis detail:", 
                                               sorted(clustered_df['cluster'].unique()))
                
                cluster_data = clustered_df[clustered_df['cluster'] == selected_cluster]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Ringkasan Klaster {selected_cluster}:**")
                    st.write(f"- Jumlah Pelanggan: {len(cluster_data)}")
                    st.write(f"- Rata-rata Pendapatan: Rp {cluster_data['total_monthly_revenue'].mean():,.0f}")
                    st.write(f"- Rata-rata Kepuasan: {cluster_data['customer_satisfaction'].mean():.1f}")
                    if 'churn' in cluster_data.columns:
                        st.write(f"- Tingkat Churn: {cluster_data['churn'].mean()*100:.1f}%")
                
                with col2:
                    st.write("**Contoh Pelanggan:**")
                    # Show sample customers from this cluster
                    sample_customers = cluster_data.head()
                    st.dataframe(sample_customers[['customer_id', 'segment', 'total_monthly_revenue', 
                                                'customer_satisfaction']], use_container_width=True)

    def explainable_ai_section(self, df):
        """Display explainable AI section"""
        st.header("🔍 Explainable AI")
        st.subheader("Model Interpretability with LIME and SHAP")
        
        # Check if models are available
        if 'model_results' not in st.session_state or 'churn_model' not in st.session_state:
            display_warning("Please train the models first in the Predictive Modeling section.")
            return
        
        results = st.session_state['model_results']
        churn_model = st.session_state['churn_model']
        
        # Library availability check
        col1, col2 = st.columns(2)
        with col1:
            if LIME_AVAILABLE:
                st.success("✅ LIME is available")
            else:
                st.error("❌ LIME not available. Install with: pip install lime")
        
        with col2:
            if SHAP_AVAILABLE:
                st.success("✅ SHAP is available")
            else:
                st.error("❌ SHAP not available. Install with: pip install shap")
        
        if not (LIME_AVAILABLE or SHAP_AVAILABLE):
            display_error("Neither LIME nor SHAP is available. Please install at least one for explainable AI.")
            return
        
        # Model selection
        available_models = list(results.keys())
        if 'test_data' in available_models:
            available_models.remove('test_data')
        
        selected_model_name = st.selectbox("Select model for explanation:", available_models)
        
        if selected_model_name and selected_model_name in results:
            model = results[selected_model_name]['model']
            test_data = results['test_data']
            X_test = test_data['X_test']
            
            # Initialize explainable AI
            explainer = ExplainableAI(model, test_data['X_test_scaled'], churn_model.feature_names)
            
            # Instance explanation
            st.subheader("🎯 Single Instance Explanation")
            
            # Dapatkan prediksi dan probabilitas
            predictions = results[selected_model_name]['predictions']
            probabilities = results[selected_model_name]['probabilities']

            # Gabungkan data prediksi dengan data asli untuk mendapatkan tanggal aktivasi
            churn_prediction_data = pd.DataFrame({
                'prediction': predictions,
                'probability': probabilities
            }, index=X_test.index)

            # Filter pelanggan yang diprediksi churn
            predicted_churn_customers = churn_prediction_data[churn_prediction_data['prediction'] == 1]

            # Gabungkan dengan DataFrame asli untuk mendapatkan 'tanggalaktivasi'
            if 'tanggalaktivasi' in df.columns:
                # Pastikan 'tanggalaktivasi' adalah datetime
                df['tanggalaktivasi'] = pd.to_datetime(df['tanggalaktivasi'], errors='coerce')
                # Gabungkan data
                predicted_churn_customers = predicted_churn_customers.join(df[['tanggalaktivasi']])
            else:
                # Jika kolom tidak ada, buat kolom kosong
                predicted_churn_customers['tanggalaktivasi'] = pd.NaT

            if not predicted_churn_customers.empty:
                st.info(f"Ditemukan {len(predicted_churn_customers)} pelanggan yang diprediksi akan churn.")

                # Opsi pengurutan
                sort_option = st.radio(
                    "Urutkan daftar pelanggan berdasarkan:",
                    ("Risiko Tertinggi", "Tanggal Aktivasi Terbaru"),
                    horizontal=True
                )

                # Urutkan data berdasarkan pilihan
                if sort_option == "Risiko Tertinggi":
                    sorted_churn_customers = predicted_churn_customers.sort_values('probability', ascending=False)
                else: # "Tanggal Aktivasi Terbaru"
                    sorted_churn_customers = predicted_churn_customers.sort_values('tanggalaktivasi', ascending=False)

                # Format pilihan dropdown untuk menampilkan ID dan probabilitas
                options = {
                    f"{idx} (Prob: {row['probability']:.2%})": idx 
                    for idx, row in sorted_churn_customers.iterrows()
                }

                selected_option = st.selectbox(
                    "Pilih Customer ID yang diprediksi Churn untuk dianalisis:",
                    options.keys()
                )
                selected_customer_id = options[selected_option]
                instance_idx = X_test.index.get_loc(selected_customer_id)
            else:
                st.warning("Tidak ada pelanggan yang diprediksi akan churn di set data tes.")
                instance_idx = st.slider("Pilih customer instance secara manual:", 0, len(X_test)-1, 0)
            
            col1, col2 = st.columns(2)
            
            # Customer info
            with col1:
                st.write("**Customer Information:**")
                
                # Ambil semua data pelanggan dari DataFrame asli menggunakan index
                customer_full_info = df.loc[selected_customer_id].copy()
                
                # Kecualikan kolom 'churn' dari tampilan jika ada
                if 'churn' in customer_full_info.index:
                    customer_full_info = customer_full_info.drop('churn')
                
                st.dataframe(customer_full_info)
                
                # Prediction
                # prediction = model.predict(test_data['X_test_scaled'][instance_idx:instance_idx+1])[0]
                # probability = model.predict_proba(test_data['X_test_scaled'][instance_idx:instance_idx+1])[0]
                
                # st.write(f"**Prediction:** {'Churn' if prediction == 1 else 'No Churn'}")
                # st.write(f"**Probability:** {probability[1]:.3f}")
            
            # Explanations
            with col2:
                if LIME_AVAILABLE:
                    if st.button("🔍 Explain with LIME"):
                        with st.spinner("Generating LIME explanation..."):
                            lime_explanation = explainer.explain_instance_lime(instance_idx, X_test)
                            
                            if lime_explanation:
                                st.success("LIME explanation generated!")
                                
                                # Display LIME explanation plot
                                try:
                                    # Ambil penjelasan untuk kelas 'Churn' (label=1)
                                    explanation_list = lime_explanation.as_list(label=1)
                                    
                                    # Daftar fitur yang akan dikecualikan dari visualisasi
                                    excluded_features = [
                                        'contract_duration_Monthly', 
                                        'segment_Basic Plan', 
                                        'service_type_Broadband'
                                    ]
                                    
                                    # Filter hanya untuk faktor pendorong churn (weight > 0) dan urutkan
                                    churn_factors = sorted([item for item in explanation_list if item[1] > 0 and not any(excluded in item[0] for excluded in excluded_features)], key=lambda x: x[1], reverse=False)
                                    
                                    # Buat plot manual hanya untuk faktor churn
                                    if churn_factors:
                                        features, weights = zip(*churn_factors)
                                        fig, ax = plt.subplots()
                                        colors = [COLOR_SCHEMES['danger']] * len(weights)
                                        ax.barh(range(len(weights)), weights, color=colors)
                                        ax.set_yticks(range(len(weights)))
                                        ax.set_yticklabels(features)
                                        ax.set_title('Faktor Pendorong ke Arah CHURN (LIME)')
                                        ax.set_xlabel('Kontribusi (Weight)')
                                    else:
                                        # Jika tidak ada faktor pendorong, buat plot kosong
                                        fig, ax = plt.subplots()
                                        ax.text(0.5, 0.5, 'Tidak ada faktor pendorong churn yang signifikan.', ha='center', va='center')
                                        ax.set_xticks([])
                                        ax.set_yticks([])

                                    fig.set_size_inches(8, 6)
                                    plt.tight_layout()
                                    st.pyplot(fig, use_container_width=True)
                                    
                                    with st.expander("Lihat Penjelasan Detail", expanded=True):
                                        # Filter hanya untuk faktor pendorong churn (weight > 0) dan urutkan
                                        churn_factors = sorted([item for item in explanation_list if item[1] > 0 and not any(excluded in item[0] for excluded in excluded_features)], key=lambda x: x[1], reverse=True)
                                        
                                        st.write("Faktor-faktor utama yang mendorong ke arah CHURN:")

                                        if churn_factors:
                                            for feature, weight in churn_factors:
                                                # Ekstrak nama fitur dari string penjelasan LIME (misal: 'tenure <= 5' -> 'tenure')
                                                feature_name = feature.split(' ')[0]
                                                
                                                # Dapatkan nilai aktual pelanggan untuk fitur tersebut
                                                actual_value = customer_full_info.get(feature_name, 'N/A')
                                                
                                                # Format nilai aktual agar lebih mudah dibaca
                                                if isinstance(actual_value, (int, float)):
                                                    if 'revenue' in feature_name or 'charges' in feature_name:
                                                        formatted_value = f"Rp {actual_value:,.0f}"
                                                    else:
                                                        formatted_value = f"{actual_value:.2f}"
                                                else:
                                                    formatted_value = str(actual_value)

                                                st.markdown(f"<li>Kondisi <b>`{feature}`</b> meningkatkan risiko churn. Nilai aktual pelanggan ini adalah <b>{formatted_value}</b>.</li>", unsafe_allow_html=True)
                                        else:
                                            st.info("Tidak ada faktor signifikan yang teridentifikasi mendorong ke arah churn untuk pelanggan ini.")
                                except Exception as e:
                                    display_error(f"Gagal menampilkan plot LIME: {e}")
                
                if SHAP_AVAILABLE:
                    if st.button("🔍 Explain with SHAP"):
                        with st.spinner("Generating SHAP explanation..."):
                            shap_values = explainer.explain_instance_shap(instance_idx, X_test)

                            # Defensive handling: SHAP libraries may return nested arrays or objects
                            try:
                                import plotly.express as px
                                import numpy as _np

                                def _normalize_shap_values(sv, instance_idx, feature_names):
                                    """Return (arr, info) where arr is 1-D numpy array or None and info is diagnostic string."""
                                    try:
                                        # Unwrap shap.Explanation-like objects
                                        if hasattr(sv, 'values'):
                                            v = _np.asarray(sv.values)
                                        elif isinstance(sv, (list, tuple)):
                                            # Often shap returns a list per class: pick positive class (index 1) if exists
                                            if len(sv) > 1:
                                                candidate = sv[1]
                                            else:
                                                candidate = sv[0]
                                            if hasattr(candidate, 'values'):
                                                v = _np.asarray(candidate.values)
                                            else:
                                                v = _np.asarray(candidate)
                                        else:
                                            v = _np.asarray(sv)
                                    except Exception as e:
                                        return None, f"extract_error:{e}"

                                    if v is None:
                                        return None, "no_data"

                                    # Normalize numpy array
                                    v = _np.asarray(v)
                                    shape = v.shape

                                    # 1-D: already good
                                    if v.ndim == 1:
                                        return v.reshape(-1), f"shape:{shape}"

                                    # 2-D: many common formats
                                    if v.ndim == 2:
                                        # If columns match features, treat rows as instances
                                        if v.shape[1] == len(feature_names):
                                            # If there are multiple rows, pick instance row
                                            if v.shape[0] > instance_idx:
                                                return v[instance_idx].reshape(-1), f"shape:{shape},selected_row"
                                            # If only one row, use it
                                            if v.shape[0] == 1:
                                                return v[0].reshape(-1), f"shape:{shape},single_row"
                                        # If rows match features (transposed), ravel
                                        if v.shape[0] == len(feature_names) and v.shape[1] == 1:
                                            return v.ravel(), f"shape:{shape},column_vector"
                                        # If rows equal number of classes, pick class 1 if possible
                                        if v.shape[0] > 1 and v.shape[0] != len(feature_names):
                                            sel = 1 if v.shape[0] > 1 else 0
                                            row = v[sel]
                                            if row.shape[0] == len(feature_names):
                                                return row.reshape(-1), f"shape:{shape},selected_class_row"
                                        # Fallback: try to ravel
                                        return v.ravel(), f"shape:{shape},ravel_fallback"

                                    # 3-D: e.g., (classes, samples, features) or (samples, classes, features)
                                    if v.ndim == 3:
                                        # Some SHAP variants return (1, n_features, n_classes) or (n_samples, n_features, n_classes)
                                        # Normalize to (samples, features, classes) possibilities and pick class index 1
                                        # Case: (1, n_features, n_classes)
                                        if v.shape[0] == 1 and v.shape[1] == len(feature_names):
                                            # v[0] is (n_features, n_classes) -> transpose to (n_classes, n_features)
                                            tmp = v[0].T  # shape (n_classes, n_features)
                                            sel_class = 1 if tmp.shape[0] > 1 else 0
                                            row = tmp[sel_class]
                                            return row.reshape(-1), f"shape:{shape},1xf_features->class{sel_class}"

                                        # Case: (n_samples, n_features, n_classes)
                                        if v.shape[0] > 1 and v.shape[1] == len(feature_names):
                                            # pick instance row and preferred class
                                            sample = v[instance_idx]
                                            # sample shape (n_features, n_classes) -> transpose
                                            if sample.ndim == 2:
                                                tmp = sample.T  # (n_classes, n_features)
                                                sel_class = 1 if tmp.shape[0] > 1 else 0
                                                return tmp[sel_class].reshape(-1), f"shape:{shape},sample_class_features_class{sel_class}"

                                        # Case: (n_classes, n_samples, n_features)
                                        if v.shape[2] == len(feature_names) and v.shape[0] > 1 and v.shape[1] > instance_idx:
                                            sel = 1 if v.shape[0] > 1 else 0
                                            return v[sel, instance_idx].reshape(-1), f"shape:{shape},class_sample_features"

                                        # fallback: ravel
                                        return v.ravel(), f"shape:{shape},ravel3"

                                    # other: ravel as last resort
                                    return v.ravel(), f"shape:{shape},ravel_default"

                                feature_names = getattr(churn_model, 'feature_names', None) or list(X_test.columns)
                                arr, info = _normalize_shap_values(shap_values, instance_idx, feature_names)

                                if arr is None:
                                    st.error(f"Unable to parse SHAP values into a 1-D array. Diagnostic: {info}. Returned type: {type(shap_values)}")
                                else:
                                    if arr.shape[0] != len(feature_names):
                                        st.error(f"SHAP values length ({arr.shape[0]}) does not match number of features ({len(feature_names)}). Diagnostic: {info}")
                                    else:
                                        st.success("SHAP explanation generated!")

                                        # Create SHAP explanation DataFrame
                                        shap_df = pd.DataFrame({
                                            'Feature': feature_names,
                                            'SHAP Value': arr
                                        })

                                        # Filter for churn factors (positive SHAP values) and sort
                                        churn_factors_df = shap_df[shap_df['SHAP Value'] > 0].sort_values('SHAP Value', ascending=False)

                                        # Display as bar chart
                                        fig = px.bar(
                                            churn_factors_df.head(10), # Show top 10 churn factors
                                            x='SHAP Value',
                                            y='Feature',
                                            orientation='h',
                                            title='Faktor Pendorong ke Arah CHURN (SHAP)',
                                            color='SHAP Value',
                                            color_continuous_scale='Reds', # Use a scale that indicates positive impact
                                            labels={'SHAP Value': 'Kontribusi terhadap Churn'}
                                        )
                                        fig.update_layout(yaxis={'categoryorder':'total ascending'})
                                        st.plotly_chart(fig, use_container_width=True)

                                        # Add explanation for the SHAP chart
                                        st.info("""
                                        **Cara Membaca Grafik SHAP:**
                                        Grafik ini menampilkan fitur-fitur yang paling kuat mendorong prediksi ke arah **Churn** untuk pelanggan yang dipilih.
                                        
                                        - **Fitur (Sumbu Y)**: Atribut pelanggan yang dianalisis.
                                        - **Kontribusi terhadap Churn (Sumbu X)**: Seberapa besar dampak positif sebuah fitur terhadap kemungkinan churn. Semakin panjang bar, semakin kuat pengaruhnya.
                                        - **Warna**: Warna yang lebih pekat menandakan nilai kontribusi yang lebih tinggi.
                                        """)

                            except Exception as e:
                                st.error(f"Error processing SHAP values: {e}")
            
            # Global explanation
            st.subheader("🌍 Global Feature Importance")
            
            if st.button("📊 Calculate Global Importance"):
                with st.spinner("Calculating global feature importance..."):
                    global_importance = explainer.get_global_feature_importance(X_test)
                    try:
                        import plotly.express as px
                        import numpy as _np

                        if global_importance is None:
                            st.error("No global feature importance returned.")
                        else:
                            feature_names = getattr(churn_model, 'feature_names', None) or list(X_test.columns)

                            # If dict, use directly
                            if isinstance(global_importance, dict):
                                items = list(global_importance.items())
                            else:
                                # Try to coerce array-like to 1-D importance vector
                                arr = _np.asarray(global_importance)

                                if arr.size == len(feature_names):
                                    flat = arr.reshape(-1)
                                elif arr.ndim >= 2 and arr.shape[-1] == len(feature_names):
                                    # take last axis as features, then if multiple entries take mean across others
                                    try:
                                        flat = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
                                    except Exception:
                                        flat = None
                                else:
                                    flat = None

                                if flat is None:
                                    st.error(f"Global importance has incompatible shape {arr.shape} for {len(feature_names)} features.")
                                    items = []
                                else:
                                    items = list(zip(feature_names, flat.tolist()))

                            if len(items) > 0:
                                importance_df = pd.DataFrame(items, columns=['Feature', 'Importance']).sort_values('Importance', ascending=False)

                                fig = px.bar(
                                    importance_df.head(15),
                                    x='Importance',
                                    y='Feature',
                                    orientation='h',
                                    title='Global Feature Importance (SHAP)',
                                    color='Importance',
                                    color_continuous_scale='Viridis'
                                )
                                st.plotly_chart(fig, use_container_width=True)

                                # Display as table
                                st.dataframe(importance_df, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error calculating global importance: {e}")

    def strategic_recommendations_section(self, df):
        """Display strategic recommendations section"""
        st.header("💡 Strategic Recommendations")
        st.subheader("Data-Driven Insights for Revenue Optimization")
        
        # Calculate key insights
        insights = self.calculate_business_insights(df)
        
        # Revenue optimization
        st.subheader("💰 Revenue Optimization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### High-Value Customer Retention")
            
            high_value_threshold = df['total_monthly_revenue'].quantile(0.8)
            high_value_customers = df[df['total_monthly_revenue'] > high_value_threshold]
            high_value_churn_rate = high_value_customers['churn'].mean() * 100 if 'churn' in df.columns else 0
            
            st.write(f"- **High-value customers:** {len(high_value_customers)} ({len(high_value_customers)/len(df)*100:.1f}%)")
            st.write(f"- **Churn rate among high-value:** {high_value_churn_rate:.1f}%")
            st.write(f"- **Potential revenue at risk:** Rp {high_value_customers[high_value_customers['churn']==1]['total_monthly_revenue'].sum():,.0f}")
            
            if high_value_churn_rate > df['churn'].mean() * 100:
                st.error("⚠️ High-value customers are churning at a higher rate!")
                st.write("**Recommended Actions:**")
                st.write("• Implement dedicated account management")
                st.write("• Offer premium support services")
                st.write("• Provide exclusive benefits and discounts")
            else:
                st.success("✅ High-value customers are well retained")
        
        with col2:
            st.markdown("#### Service Optimization")
            
            # Service performance analysis
            if 'downtime_minutes' in df.columns:
                avg_downtime = df['downtime_minutes'].mean()
                high_downtime_customers = df[df['downtime_minutes'] > avg_downtime * 1.5]
                
                st.write(f"- **Average downtime:** {avg_downtime:.0f} minutes/month")
                st.write(f"- **Customers with high downtime:** {len(high_downtime_customers)}")
                
                if 'churn' in df.columns:
                    high_downtime_churn = high_downtime_customers['churn'].mean() * 100
                    st.write(f"- **Churn rate (high downtime):** {high_downtime_churn:.1f}%")
                
                st.write("**Recommended Actions:**")
                st.write("• Invest in network infrastructure upgrades")
                st.write("• Implement proactive monitoring")
                st.write("• Establish SLA guarantees")
        
        # Customer satisfaction insights
        st.subheader("😊 Customer Satisfaction")
        
        if 'customer_satisfaction' in df.columns:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_satisfaction = df['customer_satisfaction'].mean()
                st.metric("Average Satisfaction", f"{avg_satisfaction:.1f}/10")
                
                if avg_satisfaction < 7:
                    st.error("Below target (7.5)")
                elif avg_satisfaction < 8:
                    st.warning("Needs improvement")
                else:
                    st.success("Good performance")
            
            with col2:
                low_satisfaction = df[df['customer_satisfaction'] <= 6]
                st.metric("Low Satisfaction Customers", len(low_satisfaction))
                
                if len(low_satisfaction) > 0 and 'churn' in df.columns:
                    low_sat_churn = low_satisfaction['churn'].mean() * 100
                    st.write(f"Churn rate: {low_sat_churn:.1f}%")
            
            with col3:
                high_satisfaction = df[df['customer_satisfaction'] >= 9]
                st.metric("Highly Satisfied Customers", len(high_satisfaction))
                
                if len(high_satisfaction) > 0 and 'churn' in df.columns:
                    high_sat_churn = high_satisfaction['churn'].mean() * 100
                    st.write(f"Churn rate: {high_sat_churn:.1f}%")
        
        # Segment-specific recommendations
        st.subheader("🎯 Segment-Specific Strategies")
        
        if 'segment' in df.columns:
            segments = df['segment'].unique()
            
            for segment in segments:
                segment_data = df[df['segment'] == segment]
                
                with st.expander(f"📋 {segment} Segment Strategy"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Customers:** {len(segment_data)}")
                        st.write(f"**Avg Revenue:** Rp {segment_data['total_monthly_revenue'].mean():,.0f}")
                        if 'churn' in df.columns:
                            st.write(f"**Churn Rate:** {segment_data['churn'].mean()*100:.1f}%")
                    
                    with col2:
                        # Segment-specific recommendations
                        if segment == 'Extender Package' or segment == 'Basic Plan':
                            st.write("**Focus Areas:**")
                            st.write("• Competitive pricing for basic packages")
                            st.write("• Bundle services (Internet + IPTV)")
                            st.write("• Family-oriented marketing")
                        elif segment == 'Premium Plan':
                            st.write("**Focus Areas:**")
                            st.write("• Reliable business-grade services")
                            st.write("• 24/7 technical support")
                            st.write("• Scalable bandwidth options")
                        elif segment == 'OTT Services':
                            st.write("**Focus Areas:**")
                            st.write("• Dedicated account management")
                            st.write("• Custom solutions and SLAs")
                            st.write("• Advanced security features")
                        else:
                            st.write("**Focus Areas:**")
                            st.write("• Compliance with regulations")
                            st.write("• High security standards")
                            st.write("• Long-term contract incentives")
        
        # Action plan
        st.subheader("🚀 Action Plan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Immediate Actions (0-3 months)")
            st.write("1. **Implement churn prediction system**")
            st.write("   - Deploy trained ML models")
            st.write("   - Set up automated alerts")
            st.write("")
            st.write("2. **Launch retention campaigns**")
            st.write("   - Target high-risk customers")
            st.write("   - Offer personalized incentives")
            st.write("")
            st.write("3. **Improve customer service**")
            st.write("   - Reduce response times")
            st.write("   - Enhance technical support")
        
        with col2:
            st.markdown("#### Medium-term Actions (3-12 months)")
            st.write("1. **Network infrastructure upgrades**")
            st.write("   - Reduce downtime incidents")
            st.write("   - Improve service quality")
            st.write("")
            st.write("2. **Product portfolio optimization**")
            st.write("   - Develop new service packages")
            st.write("   - Adjust pricing strategies")
            st.write("")
            st.write("3. **Customer segmentation implementation**")
            st.write("   - Targeted marketing campaigns")
            st.write("   - Personalized service offerings")
        
        # ROI calculation
        st.subheader("💹 Expected ROI")
        
        if 'churn' in df.columns and 'total_monthly_revenue' in df.columns:
            current_churn_rate = df['churn'].mean()
            monthly_revenue_at_risk = df[df['churn'] == 1]['total_monthly_revenue'].sum()
            
            # Assume 25% reduction in churn with interventions
            improved_churn_rate = current_churn_rate * 0.75
            revenue_saved = monthly_revenue_at_risk * 0.25
            annual_revenue_saved = revenue_saved * 12
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Monthly Revenue at Risk", f"Rp {monthly_revenue_at_risk:,.0f}")
            
            with col2:
                st.metric("Potential Monthly Savings", f"Rp {revenue_saved:,.0f}")
            
            with col3:
                st.metric("Estimated Annual ROI", f"Rp {annual_revenue_saved:,.0f}")
            
            st.info(f"📊 **Assumptions:** 25% reduction in churn rate through targeted interventions")

        # =============================================
        # Individual Customer Recommendations
        # =============================================
        st.subheader("🎯 Rekomendasi per Pelanggan (Actionable Insights)")

        # Check if models are available
        if 'model_results' not in st.session_state or 'churn_model' not in st.session_state:
            display_warning("Latih model terlebih dahulu di bagian 'Predictive Modeling' untuk mendapatkan rekomendasi per pelanggan.")
            return

        results = st.session_state['model_results']
        test_data = results.get('test_data', {})
        X_test = test_data.get('X_test')

        # Get predictions from the best available model
        if 'random_forest' in results:
            predictions = results['random_forest']['predictions']
        elif 'xgboost' in results:
            predictions = results['xgboost']['predictions']
        else:
            display_error("Tidak ada hasil prediksi yang ditemukan.")
            return

        # Find customers predicted to churn
        probabilities = results['random_forest']['probabilities'] if 'random_forest' in results else results['xgboost']['probabilities']        
        churn_prediction_data = pd.DataFrame({
            'prediction': predictions,
            'probability': probabilities
        }, index=X_test.index)

        # Filter pelanggan yang diprediksi churn
        predicted_churn_customers = churn_prediction_data[churn_prediction_data['prediction'] == 1]

        # Gabungkan dengan DataFrame asli untuk mendapatkan 'tanggalaktivasi'
        if 'tanggalaktivasi' in df.columns:
            df['tanggalaktivasi'] = pd.to_datetime(df['tanggalaktivasi'], errors='coerce')
            predicted_churn_customers = predicted_churn_customers.join(df[['tanggalaktivasi']])
        else:
            predicted_churn_customers['tanggalaktivasi'] = pd.NaT

        if predicted_churn_customers.empty:
            st.success("✅ Tidak ada pelanggan yang diprediksi akan churn pada data tes saat ini.")
            return

        st.info(f"Ditemukan {len(predicted_churn_customers)} pelanggan berisiko churn. Pilih satu untuk melihat rekomendasi tindakan.")

        # Opsi pengurutan
        sort_option_rec = st.radio(
            "Urutkan daftar pelanggan berdasarkan:",
            ("Risiko Tertinggi", "Tanggal Aktivasi Terbaru"),
            horizontal=True,
            key="recommendation_sort" # Kunci unik untuk widget ini
        )

        # Urutkan data berdasarkan pilihan
        if sort_option_rec == "Risiko Tertinggi":
            sorted_churn_customers = predicted_churn_customers.sort_values('probability', ascending=False)
        else: # "Tanggal Aktivasi Terbaru"
            sorted_churn_customers = predicted_churn_customers.sort_values('tanggalaktivasi', ascending=False)


        # Format pilihan dropdown untuk menampilkan ID dan probabilitas
        options = {
            f"{idx} (Prob: {row['probability']:.2%})": idx 
            for idx, row in sorted_churn_customers.iterrows()
        }
        
        selected_option = st.selectbox(
            "Pilih Customer ID yang berisiko Churn untuk dianalisis:",
            options.keys(),
            key="recommendation_select" # Kunci unik untuk widget ini
        )
        
        if not selected_option:
            return
            
        selected_customer_id = options[selected_option]

        if selected_customer_id:
            customer_data = df.loc[selected_customer_id]
            
            col1, col2 = st.columns([1, 2])

            with col1:
                st.write("**Informasi Pelanggan:**")
                customer_info = customer_data.drop('churn') if 'churn' in customer_data.index else customer_data
                st.dataframe(customer_info)

            with col2:
                st.write("**Rekomendasi Tindakan:**")
                recommendations = self.generate_individual_recommendations(customer_data)
                if recommendations:
                    for rec in recommendations:
                        st.markdown(f"- {rec}")
                else:
                    st.write("Tidak ada rekomendasi spesifik yang dapat dibuat untuk pelanggan ini.")

    def generate_individual_recommendations(self, customer_data: pd.Series) -> list:
        """Generate specific recommendations for a single customer."""
        recs = []
        if customer_data.get('complaint_count', 0) > 1:
            recs.append(f"**Hubungi Pelanggan Proaktif**: Lakukan panggilan untuk menindaklanjuti {int(customer_data['complaint_count'])} keluhan terakhir dan pastikan masalah telah terselesaikan.")
        if customer_data.get('tenure', 12) <= 6:
            recs.append("**Tawarkan Bonus Loyalitas**: Berikan diskon khusus atau bonus kuota untuk 3 bulan ke depan sebagai apresiasi pelanggan baru.")
        if customer_data.get('downtime_minutes', 0) > 120:
            recs.append(f"**Berikan Kompensasi Layanan**: Tawarkan kompensasi (misal: potongan tagihan) untuk `downtime` tinggi ({int(customer_data['downtime_minutes'])} menit) bulan lalu.")
        if customer_data.get('contract_duration') == 'Monthly':
            recs.append("**Tawarkan Upgrade Kontrak**: Berikan diskon menarik untuk beralih dari kontrak bulanan ke kontrak 1 atau 2 tahun untuk meningkatkan loyalitas.")
        if customer_data.get('customer_satisfaction', 10) <= 6:
            recs.append(f"**Lakukan Survei Kepuasan**: Kirim survei singkat untuk memahami alasan di balik skor kepuasan yang rendah ({int(customer_data['customer_satisfaction'])}/10) dan tawarkan solusi.")
        if customer_data.get('payment_delay_days', 0) > 7:
            recs.append("**Tawarkan Opsi Pembayaran**: Informasikan mengenai opsi pembayaran yang lebih mudah atau berikan keringanan untuk pembayaran yang tertunda.")
        
        if not recs:
            recs.append("Pelanggan ini tidak menunjukkan sinyal masalah yang jelas. Pertimbangkan untuk menawarkan promosi umum atau menanyakan kepuasan layanan secara langsung.")
            
        return recs

    def calculate_business_insights(self, df):
        """Calculate business insights from data"""
        insights = {}
        
        if 'total_monthly_revenue' in df.columns:
            insights['total_revenue'] = df['total_monthly_revenue'].sum()
            insights['avg_revenue'] = df['total_monthly_revenue'].mean()
            insights['revenue_std'] = df['total_monthly_revenue'].std()
        
        if 'churn' in df.columns:
            insights['churn_rate'] = df['churn'].mean()
            insights['churned_customers'] = df['churn'].sum()
            insights['retention_rate'] = 1 - insights['churn_rate']
        
        if 'customer_satisfaction' in df.columns:
            insights['avg_satisfaction'] = df['customer_satisfaction'].mean()
            insights['low_satisfaction_customers'] = len(df[df['customer_satisfaction'] <= 6])
        
        return insights

    def model_management_section(self, df):
        """Display model management section"""
        st.header("⚙️ Model Management")
        st.subheader("Save, Load, and Export Models")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Model Operations")
            
            # Check if models exist in session
            if 'model_results' in st.session_state:
                st.success("✅ Models available in memory")
                
                if st.button("💾 Save Models to Disk"):
                    churn_model = st.session_state.get('churn_model')
                    if churn_model:
                        churn_model.save_models(st.session_state['model_results'])
                
                if st.button("📤 Export Model Performance Report"):
                    self.export_model_report(st.session_state['model_results'])
            else:
                st.warning("⚠️ No models in memory. Train models first.")
            
            # Load models from disk
            if st.button("📂 Load Models from Disk"):
                churn_model = ChurnPredictionModel()
                if churn_model.load_models():
                    st.session_state['churn_model'] = churn_model
                    display_success("Models loaded successfully!")
        
        with col2:
            st.markdown("#### Data Export")
            
            if st.button("📊 Export Current Dataset"):
                self.export_data(df)
            
            if 'segmentation_results' in st.session_state:
                if st.button("👥 Export Segmented Customers"):
                    segmented_df = st.session_state['segmentation_results']['clustered_data']
                    self.export_data(segmented_df, "segmented_customers")
        
        # Model versioning
        st.subheader("📋 Model Information")
        
        if 'model_results' in st.session_state:
            results = st.session_state['model_results']
            
            model_info = {
                "Random Forest Available": "Yes" if 'random_forest' in results else "No",
                "XGBoost Available": "Yes" if 'xgboost' in results else "No",
                "Training Date": "Current Session",
                "Feature Count": len(st.session_state.get('churn_model', ChurnPredictionModel()).feature_names or [])
            }
            
            info_df = pd.DataFrame(list(model_info.items()), columns=['Property', 'Value'])
            st.dataframe(info_df, use_container_width=True)

    def export_model_report(self, results):
        """Export model performance report"""
        try:
            report_data = []
            
            for model_name, model_results in results.items():
                if model_name != 'test_data' and 'classification_report' in model_results:
                    report = model_results['classification_report']
                    
                    report_data.append({
                        'Model': model_name.title(),
                        'Accuracy': report['accuracy'],
                        'Precision (Churn)': report['1']['precision'],
                        'Recall (Churn)': report['1']['recall'],
                        'F1-Score (Churn)': report['1']['f1-score']
                    })
            
            if report_data:
                report_df = pd.DataFrame(report_data)
                csv = report_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Model Report",
                    data=csv,
                    file_name="iconnet_model_performance_report.csv",
                    mime="text/csv"
                )
                
                display_success("Model report ready for download!")
        
        except Exception as e:
            display_error(f"Error exporting model report: {str(e)}")

    def export_data(self, df, filename_prefix="iconnet_data"):
        """Export data to CSV"""
        try:
            csv = df.to_csv(index=False)
            
            st.download_button(
                label=f"📥 Download {filename_prefix.replace('_', ' ').title()}",
                data=csv,
                file_name=f"{filename_prefix}.csv",
                mime="text/csv"
            )
            
            display_success("Data ready for download!")
        
        except Exception as e:
            display_error(f"Error exporting data: {str(e)}")

    def run(self):
        """Run the main dashboard"""
        # Display header and intro
        # self.display_header()
        self.display_team_info()
        
        # Main title
        st.title("📊 ICONNET Predictive Analytics Dashboard")
        
        # Description
        st.markdown("""
        <div class="highlight">
        Dashboard ini mendemonstrasikan bagaimana analitik prediktif dan algoritma machine learning
        (Random Forest, XGBoost, dan K-Means Clustering) dapat mengoptimalkan pendapatan layanan ICONNET
        dan meningkatkan efisiensi operasional di PT PLN ICON PLUS SBU Regional Jawa Tengah.
        </div>
        """, unsafe_allow_html=True)
        
        # Load data
        df = self.load_data_section()
        
        if df is None:
            st.warning("⚠️ Tidak ada data yang dimuat. Silakan upload file CSV atau pilih dari data yang tersimpan.")
            st.info("💡 Gunakan sidebar untuk memilih sumber data.")
            return
        
        # Navigation
        st.sidebar.header("🚀 ICONNET Analytics Navigation")
        section = st.sidebar.radio("Pilih Section:", DASHBOARD_SECTIONS)
        
        # Add info box in sidebar
        st.sidebar.markdown("""
        <div class="info-box">
        <strong>💡 Tip:</strong><br>
        Gunakan navigasi di samping untuk menjelajahi berbagai analisis prediktif yang tersedia.
        </div>
        """, unsafe_allow_html=True)
        
        # Display selected section
        try:
            if section == "Data Overview":
                self.data_overview_section(df)
            elif section == "Churn Analysis":
                self.churn_analysis_section(df)
            elif section == "Predictive Modeling":
                self.predictive_modeling_section(df)
            elif section == "Customer Segmentation":
                self.customer_segmentation_section(df)
            elif section == "Explainable AI":
                self.explainable_ai_section(df)
            elif section == "Strategic Recommendations":
                self.strategic_recommendations_section(df)
            elif section == "Model Management":
                self.model_management_section(df)
        
        except Exception as e:
            display_error(f"Error in {section}: {str(e)}")
            logger.error(f"Section error in {section}: {e}")
        
        # Footer
        st.markdown("""
        <div class="footer">
        ICONNET Predictive Analytics Dashboard - Dibuat untuk PT PLN ICON PLUS SBU Regional Jawa Tengah<br>
        Dashboard ini menggunakan teknologi Machine Learning untuk optimalisasi pendapatan dan retensi pelanggan.
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main function to run the dashboard"""
    try:
        dashboard = IconnetDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Critical error in dashboard: {str(e)}")
        st.info("Please refresh the page and try again.")

if __name__ == "__main__":
    main()