# -*- coding: utf-8 -*-
"""
Visualization module for ICONNET Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
from utils import get_color_palette, format_currency, format_percentage, handle_error
from config import COLOR_SCHEMES

class DashboardVisualizations:
    """Create visualizations for the dashboard"""
    
    def __init__(self):
        self.color_palette = get_color_palette(10)
        
    @handle_error
    def create_revenue_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create revenue distribution histogram"""
        fig = px.histogram(
            df, 
            x='total_monthly_revenue',
            title='Monthly Revenue Distribution',
            nbins=30,
            color_discrete_sequence=[COLOR_SCHEMES["primary"]]
        )
        
        fig.update_layout(
            showlegend=False,
            xaxis_title="Monthly Revenue (Rp)",
            yaxis_title="Number of Customers",
            template="plotly_white"
        )
        
        return fig
    
    @handle_error
    def create_segment_pie_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create customer segments pie chart"""
        segment_counts = df['segment'].value_counts()
        
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title='Customer Segments Distribution',
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig
    
    @handle_error
    def create_contract_bar_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create contract duration bar chart"""
        contract_counts = df['contract_duration'].value_counts()
        
        fig = px.bar(
            x=contract_counts.index,
            y=contract_counts.values,
            title='Contract Duration Distribution',
            color_discrete_sequence=[COLOR_SCHEMES["secondary"]]
        )
        
        fig.update_layout(
            xaxis_title="Contract Duration",
            yaxis_title="Number of Customers",
            template="plotly_white"
        )
        
        return fig
    
    @handle_error
    def create_churn_analysis_charts(self, df: pd.DataFrame) -> Dict[str, go.Figure]:
        """Create churn analysis visualizations"""
        charts = {}
        
        # Churn by segment
        churn_by_segment = df.groupby('segment')['churn'].agg(['mean', 'count', 'sum']).reset_index()
        churn_by_segment.columns = ['segment', 'churn_rate', 'total_customers', 'churned_customers']
        
        charts['churn_by_segment'] = px.bar(
            churn_by_segment,
            x='segment',
            y='churn_rate',
            title='Churn Rate by Customer Segment',
            color='churn_rate',
            color_continuous_scale='Reds'
        )
        
        # Churn by service type
        churn_by_service = df.groupby('service_type')['churn'].agg(['mean', 'count', 'sum']).reset_index()
        churn_by_service.columns = ['service_type', 'churn_rate', 'total_customers', 'churned_customers']
        
        charts['churn_by_service'] = px.bar(
            churn_by_service,
            x='service_type',
            y='churn_rate',
            title='Churn Rate by Service Type',
            color='churn_rate',
            color_continuous_scale='Blues'
        )
        
        # Box plots for churn factors
        charts['tenure_vs_churn'] = px.box(
            df, x='churn', y='tenure',
            title='Tenure vs Churn',
            color='churn',
            color_discrete_map={0: COLOR_SCHEMES["success"], 1: COLOR_SCHEMES["danger"]}
        )
        
        charts['satisfaction_vs_churn'] = px.box(
            df, x='churn', y='customer_satisfaction',
            title='Customer Satisfaction vs Churn',
            color='churn',
            color_discrete_map={0: COLOR_SCHEMES["success"], 1: COLOR_SCHEMES["danger"]}
        )
        
        charts['downtime_vs_churn'] = px.box(
            df, x='churn', y='downtime_minutes',
            title='Downtime vs Churn',
            color='churn',
            color_discrete_map={0: COLOR_SCHEMES["success"], 1: COLOR_SCHEMES["danger"]}
        )
        
        # Update x-axis labels for box plots
        for chart_name in ['tenure_vs_churn', 'satisfaction_vs_churn', 'downtime_vs_churn']:
            charts[chart_name].update_xaxes(tickvals=[0, 1], ticktext=['No Churn', 'Churn'])
            
        return charts
    
    @handle_error
    def create_correlation_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title='Feature Correlation Matrix',
            color_continuous_scale='RdBu',
            color_continuous_midpoint=0
        )
        
        return fig
    
    @handle_error
    def create_confusion_matrix(self, cm: np.ndarray, title: str) -> go.Figure:
        """Create confusion matrix heatmap"""
        fig = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Not Churn', 'Churn'],
            y=['Not Churn', 'Churn'],
            title=title,
            color_continuous_scale='Blues'
        )
        
        return fig
    
    @handle_error
    def create_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray, model_name: str, roc_auc_value: Optional[float] = None) -> go.Figure:
        """Create ROC curve"""
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        
        # Use provided roc_auc_value if available, otherwise calculate it
        if roc_auc_value is None:
            roc_auc = auc(fpr, tpr)
        else:
            roc_auc = roc_auc_value
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{model_name}',
            line=dict(color=COLOR_SCHEMES["primary"], width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title=f'<b>ROC Curve - {model_name}</b>',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            template="plotly_white",
            legend=dict(x=0.6, y=0.1, traceorder="normal")
        )
        
        # Menambahkan anotasi untuk skor AUC
        fig.add_annotation(
            x=0.95, y=0.05,
            text=f"<b>AUC = {roc_auc:.4f}</b>",
            showarrow=False,
            font=dict(size=14, color="black"),
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
        
        return fig
    
    @handle_error
    def create_precision_recall_curve(self, y_true: np.ndarray, y_scores: np.ndarray, model_name: str) -> go.Figure:
        """Create Precision-Recall curve"""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'{model_name} (AP = {avg_precision:.2f})',
            line=dict(color=COLOR_SCHEMES["secondary"], width=2)
        ))
        
        fig.update_layout(
            title=f'Precision-Recall Curve - {model_name}',
            xaxis_title='Recall',
            yaxis_title='Precision',
            template="plotly_white"
        )
        
        return fig
    
    @handle_error
    def create_feature_importance_chart(self, feature_importance: Dict[str, float], title: str) -> go.Figure:
        """Create feature importance bar chart"""
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        features, importance = zip(*sorted_features[:15])  # Top 15 features
        
        fig = px.bar(
            x=list(importance),
            y=list(features),
            orientation='h',
            title=title,
            color_discrete_sequence=[COLOR_SCHEMES["primary"]]
        )
        
        fig.update_layout(
            xaxis_title="Importance",
            yaxis_title="Features",
            template="plotly_white"
        )
        
        return fig
    
    @handle_error
    def create_shap_global_summary(self, shap_values: np.ndarray, feature_names: list, title: str = "Global Feature Importance (SHAP)") -> go.Figure:
        """Create SHAP global summary plot"""
        try:
            # Calculate mean absolute SHAP values
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
            # Create DataFrame for plotting
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'SHAP_Importance': mean_abs_shap
            }).sort_values('SHAP_Importance', ascending=True)  # Sort for horizontal bar plot
            
            # Take top 15 features
            feature_importance_df = feature_importance_df.tail(15)
            
            # Create plot
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=feature_importance_df['Feature'],
                x=feature_importance_df['SHAP_Importance'],
                orientation='h',
                marker=dict(
                    color=feature_importance_df['SHAP_Importance'],
                    colorscale='RdBu_r',
                    showscale=True,
                    colorbar=dict(title="SHAP Value")
                ),
                hovertemplate='<b>%{y}</b><br>SHAP Importance: %{x:.4f}<extra></extra>'
            ))
            
            # Highlight top 3 features
            top_3_features = feature_importance_df.tail(3)
            for i, (idx, row) in enumerate(top_3_features.iterrows()):
                fig.add_annotation(
                    x=row['SHAP_Importance'] + 0.01,
                    y=row['Feature'],
                    text=f"#{3-i}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="red",
                    font=dict(color="red", size=12, weight="bold"),
                    bgcolor="white",
                    bordercolor="red",
                    borderwidth=1
                )
            
            fig.update_layout(
                title=dict(
                    text=f"<b>{title}</b><br>Top Factors Driving Customer Churn",
                    x=0.5,
                    xanchor='center',
                    font=dict(size=16, color='#1f77b4')
                ),
                xaxis_title="Mean |SHAP Value| (Impact on Model Output)",
                yaxis_title="Features",
                template="plotly_white",
                height=600,
                showlegend=False,
                margin=dict(l=100, r=50, t=100, b=50)
            )
            
            return fig
            
        except Exception as e:
            # Fallback to simple bar chart if SHAP processing fails
            st.error(f"Error creating SHAP plot: {e}")
            return self.create_feature_importance_chart(
                dict(zip(feature_names, np.mean(np.abs(shap_values), axis=0))),
                title
            )
    
    @handle_error
    def create_shap_waterfall_plot(self, shap_values: np.ndarray, feature_names: list, instance_idx: int, 
                                 base_value: float, prediction: float, title: str = "SHAP Waterfall Plot") -> go.Figure:
        """Create SHAP waterfall plot for individual instance"""
        try:
            # Get SHAP values for this instance
            if len(shap_values.shape) == 3:
                # Handle 3D array (common in SHAP output)
                instance_shap = shap_values[instance_idx, :, 1]  # Class 1 (churn)
            else:
                instance_shap = shap_values[instance_idx, :]
            
            # Create DataFrame
            shap_df = pd.DataFrame({
                'Feature': feature_names,
                'SHAP_Value': instance_shap
            }).sort_values('SHAP_Value', key=abs, ascending=False)
            
            # Take top 10 features
            shap_df = shap_df.head(10).sort_values('SHAP_Value', ascending=True)
            
            # Create waterfall data
            features = shap_df['Feature'].tolist()
            values = shap_df['SHAP_Value'].tolist()
            
            # Add base value and prediction
            features = ['Base Value'] + features + ['Prediction']
            values = [base_value] + values + [prediction]
            
            fig = go.Figure(go.Waterfall(
                name="SHAP",
                orientation="v",
                measure=["absolute"] + ["relative"] * (len(features)-2) + ["total"],
                x=features,
                y=values,
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            ))
            
            fig.update_layout(
                title=dict(
                    text=f"<b>{title}</b><br>Feature Contributions to Prediction",
                    x=0.5,
                    xanchor='center'
                ),
                xaxis_title="Features",
                yaxis_title="SHAP Value",
                template="plotly_white",
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating SHAP waterfall plot: {e}")
            return None
    
    @handle_error
    def create_cluster_visualization(self, df: pd.DataFrame, cluster_col: str = 'cluster') -> Dict[str, go.Figure]:
        """Create cluster visualization charts"""
        charts = {}
        
        # Cluster distribution pie chart
        cluster_counts = df[cluster_col].value_counts().sort_index()
        charts['cluster_distribution'] = px.pie(
            values=cluster_counts.values,
            names=[f'Cluster {i}' for i in cluster_counts.index],
            title='Customer Cluster Distribution',
            color_discrete_sequence=self.color_palette
        )
        
        # Revenue by cluster
        revenue_by_cluster = df.groupby(cluster_col)['total_monthly_revenue'].agg(['mean', 'sum']).reset_index()
        charts['revenue_by_cluster'] = px.bar(
            revenue_by_cluster,
            x=cluster_col,
            y='mean',
            title='Average Revenue by Cluster',
            color_discrete_sequence=[COLOR_SCHEMES["primary"]]
        )
        
        # Scatter plot: Revenue vs Satisfaction colored by cluster
        charts['revenue_vs_satisfaction'] = px.scatter(
            df,
            x='customer_satisfaction',
            y='total_monthly_revenue',
            color=cluster_col,
            title='Revenue vs Customer Satisfaction by Cluster',
            color_discrete_sequence=self.color_palette
        )
        
        # Churn rate by cluster (if churn column exists)
        if 'churn' in df.columns:
            churn_by_cluster = df.groupby(cluster_col)['churn'].mean().reset_index()
            charts['churn_by_cluster'] = px.bar(
                churn_by_cluster,
                x=cluster_col,
                y='churn',
                title='Churn Rate by Cluster',
                color_discrete_sequence=[COLOR_SCHEMES["danger"]]
            )
        
        return charts
    
    @handle_error
    def create_time_series_chart(self, df: pd.DataFrame, date_col: str, value_col: str, title: str) -> go.Figure:
        """Create time series chart"""
        fig = px.line(
            df,
            x=date_col,
            y=value_col,
            title=title,
            color_discrete_sequence=[COLOR_SCHEMES["primary"]]
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=value_col.replace('_', ' ').title(),
            template="plotly_white"
        )
        
        return fig
    
    @handle_error
    def create_metrics_dashboard(self, metrics: Dict[str, Any]) -> None:
        """Create metrics dashboard with cards"""
        cols = st.columns(len(metrics))
        
        for i, (metric_name, metric_data) in enumerate(metrics.items()):
            with cols[i]:
                if isinstance(metric_data, dict):
                    value = metric_data.get('value', 0)
                    delta = metric_data.get('delta', None)
                    format_type = metric_data.get('format', 'number')
                else:
                    value = metric_data
                    delta = None
                    format_type = 'number'
                
                # Format value based on type
                if format_type == 'currency':
                    formatted_value = format_currency(value)
                elif format_type == 'percentage':
                    formatted_value = format_percentage(value)
                else:
                    formatted_value = f"{value:,.0f}" if isinstance(value, (int, float)) else str(value)
                
                st.metric(
                    label=metric_name.replace('_', ' ').title(),
                    value=formatted_value,
                    delta=delta
                )
    
    @handle_error
    def create_comparison_chart(self, models_results: Dict[str, Any], metric: str = 'accuracy') -> go.Figure:
        """Create model comparison chart"""
        model_names = []
        metric_values = []
        
        for model_name, results in models_results.items():
            if 'classification_report' in results:
                model_names.append(model_name.title())
                metric_values.append(results['classification_report'][metric])
        
        fig = px.bar(
            x=model_names,
            y=metric_values,
            title=f'Model Comparison - {metric.title()}',
            color_discrete_sequence=[COLOR_SCHEMES["primary"]]
        )
        
        fig.update_layout(
            xaxis_title="Models",
            yaxis_title=metric.title(),
            template="plotly_white"
        )
        
        return fig
    
    @handle_error
    def create_shap_dependence_plot(self, shap_values: np.ndarray, features: np.ndarray, 
                                  feature_names: list, target_feature: str, 
                                  title: str = "SHAP Dependence Plot") -> go.Figure:
        """Create SHAP dependence plot"""
        try:
            # Find target feature index
            target_idx = feature_names.index(target_feature)
            
            # Get SHAP values and feature values for target feature
            shap_target = shap_values[:, target_idx]
            feature_target = features[:, target_idx]
            
            fig = px.scatter(
                x=feature_target,
                y=shap_target,
                title=f"{title}<br>{target_feature} vs SHAP Value",
                labels={'x': target_feature, 'y': 'SHAP Value'},
                trendline="lowess"
            )
            
            fig.update_layout(
                template="plotly_white",
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating SHAP dependence plot: {e}")
            return None