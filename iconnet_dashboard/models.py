# -*- coding: utf-8 -*-
"""
Machine Learning models module for ICONNET Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score, silhouette_score
)
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Import Plotly untuk visualisasi
import plotly.express as px
import plotly.graph_objects as go

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from config import MODEL_CONFIG, MODELS_DIR
from utils import handle_error, display_success, display_warning, display_error, logger

class ChurnPredictionModel:
    """Churn prediction model using Random Forest and XGBoost"""
    
    def __init__(self):
        self.rf_model = None
        self.xgb_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.models_dir = MODELS_DIR
        
    @handle_error
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for modeling"""
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")
        
        model_df = df.copy()
        
        # Clean currency and numeric columns before processing
        numeric_cols_to_clean = ['monthly_charges', 'total_monthly_revenue']
        for col in numeric_cols_to_clean:
            if col in model_df.columns and model_df[col].dtype == 'object':
                # Convert to string, remove non-digit characters, then convert to numeric
                model_df[col] = model_df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
                model_df[col] = pd.to_numeric(model_df[col], errors='coerce').fillna(0)

        # Encode categorical variables
        categorical_cols = ['segment', 'contract_duration', 'service_type', 'additional_services', 'payment_method']
        existing_categorical = [col for col in categorical_cols if col in model_df.columns]
        
        if existing_categorical:
            model_df = pd.get_dummies(model_df, columns=existing_categorical, prefix=existing_categorical)
        
        # Define features and target
        exclude_cols = ['customer_id', 'churn', 'total_monthly_revenue','namapelanggan', 'email', 'telepon', 'alamat', 'ALAMAT', 'layanan', 'service_type', 'segment','namakp','KABUPATEN', 'tanggalaktivasi']
        feature_cols = [col for col in model_df.columns if col not in exclude_cols]
        
        X = model_df[feature_cols]
        y = model_df['churn'] if 'churn' in model_df.columns else None
        
        # Store feature names
        self.feature_names = feature_cols
        
        logger.info(f"Data prepared: {len(X)} samples, {len(feature_cols)} features")
        return X, y
    
    @handle_error
    def handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Handle class imbalance using cluster-based undersampling"""
        if y is None:
            return X, y
            
        # Separate majority and minority classes
        majority_class_idx = y[y == 0].index
        minority_class_idx = y[y == 1].index
        
        # If there's no minority class, or no majority class, return original data
        # The check in train_models will then catch if y_train (now y_balanced) has only one class.
        if len(minority_class_idx) == 0:
            logger.warning("No minority class (churn=1) found in training set. Skipping imbalance handling.")
            return X, y
        if len(majority_class_idx) == 0:
            logger.warning("No majority class (churn=0) found in training set. Skipping imbalance handling.")
            return X, y

        # Only undersample if majority is significantly larger than minority
        if len(majority_class_idx) <= len(minority_class_idx):
            logger.info("Majority class is not larger than minority. Skipping undersampling.")
            return X, y

        majority_class = X.loc[majority_class_idx]

        # Determine n_clusters more robustly.
        # It should not exceed the number of minority samples (which is the target size for the undersampled majority)
        # and also not exceed the number of majority samples.
        n_clusters = min(5, len(minority_class_idx), len(majority_class))
        if n_clusters < 2: # K-Means needs at least 2 clusters to make sense for sampling strategy
            logger.warning(f"Not enough samples for effective clustering ({n_clusters} clusters). Skipping imbalance handling.")
            return X, y

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

        try:
            majority_clusters = kmeans.fit_predict(majority_class)

            sampled_indices = []
            # samples_per_cluster should be such that total majority samples roughly equals minority samples
            samples_per_cluster = max(1, int(np.ceil(len(minority_class_idx) / n_clusters)))

            for cluster_id in range(n_clusters):
                cluster_mask = majority_clusters == cluster_id
                cluster_indices = majority_class.index[cluster_mask]

                if len(cluster_indices) > 0:
                    sample_size = min(len(cluster_indices), samples_per_cluster)
                    sampled = np.random.choice(cluster_indices, sample_size, replace=False)
                    sampled_indices.extend(sampled)

            # Create balanced dataset
            balanced_indices = list(sampled_indices) + list(minority_class_idx)
            X_balanced = X.loc[balanced_indices]
            y_balanced = y.loc[balanced_indices]
            
            logger.info(f"Class imbalance handled: {len(X_balanced)} balanced samples ({len(sampled_indices)} majority, {len(minority_class_idx)} minority)")
            return X_balanced, y_balanced
            
        except Exception as e:
            logger.warning(f"Cluster-based sampling failed: {e}. Using original data.")
            return X, y
    
    @st.cache_data(ttl=1800)  # Cache for 30 minutes
    def train_models(_self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train Random Forest and XGBoost models"""
        if X is None or y is None:
            raise ValueError("Training data is None")
        
        # --- NEW: Initial validation of the target variable 'y' ---
        # Check if the entire dataset's target column has both classes before splitting.
        if len(y.unique()) < 2:
            class_counts = y.value_counts()
            return {"error": f"The target column 'churn' only contains one class: {class_counts.to_dict()}. To train a predictive model, the dataset must contain historical examples of both churn (1) and non-churn (0) customers."}

        results = {}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # --- REVISED VALIDATION STEP: Check for presence of both classes in y_train ---
        # This is crucial because stratification can fail if the minority class has very few samples.
        if len(y_train.unique()) < 2:
            class_counts = y_train.value_counts()
            return {"error": f"The training set does not contain both classes after splitting. It has: {class_counts.to_dict()}. Cannot train a reliable model. This usually happens with extreme class imbalance."}


        # Handle class imbalance
        X_train_balanced, y_train_balanced = _self.handle_class_imbalance(X_train, y_train)
        
        # NEW VALIDATION STEP: Check if y_train_balanced has both classes
        if len(np.unique(y_train_balanced)) < 2:
            return {"error": "The balanced training set contains only one class. Cannot train the model effectively. Consider adjusting class imbalance handling or using a larger dataset."}


        # Scale features
        X_train_scaled = _self.scaler.fit_transform(X_train_balanced)
        X_test_scaled = _self.scaler.transform(X_test)
        
        # Train Random Forest
        logger.info("Training Random Forest model...")
        _self.rf_model = RandomForestClassifier(**MODEL_CONFIG["random_forest"])
        _self.rf_model.fit(X_train_scaled, y_train_balanced)
        
        # Predictions
        y_pred_rf = _self.rf_model.predict(X_test_scaled)
        y_pred_proba_rf = _self.rf_model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        rf_report = classification_report(y_test, y_pred_rf, output_dict=True)
        rf_cm = confusion_matrix(y_test, y_pred_rf)
        rf_roc_auc = roc_auc_score(y_test, y_pred_proba_rf)
        
        results['random_forest'] = {
            'model': _self.rf_model,
            'predictions': y_pred_rf,
            'probabilities': y_pred_proba_rf,
            'classification_report': rf_report,
            'roc_auc': rf_roc_auc,
            'confusion_matrix': rf_cm,
            'feature_importance': dict(zip(_self.feature_names, _self.rf_model.feature_importances_))
        }
        
        # Train XGBoost if available
        if XGBOOST_AVAILABLE:
            logger.info("Training XGBoost model...")
            _self.xgb_model = XGBClassifier(**MODEL_CONFIG["xgboost"])
            _self.xgb_model.fit(X_train_scaled, y_train_balanced)
            
            # Predictions
            y_pred_xgb = _self.xgb_model.predict(X_test_scaled)
            y_pred_proba_xgb = _self.xgb_model.predict_proba(X_test_scaled)[:, 1]
            
            # Metrics
            xgb_report = classification_report(y_test, y_pred_xgb, output_dict=True)
            xgb_cm = confusion_matrix(y_test, y_pred_xgb)
            xgb_roc_auc = roc_auc_score(y_test, y_pred_proba_xgb)
            
            results['xgboost'] = {
                'model': _self.xgb_model,
                'predictions': y_pred_xgb,
                'probabilities': y_pred_proba_xgb,
                'classification_report': xgb_report,
                'roc_auc': xgb_roc_auc,
                'confusion_matrix': xgb_cm,
                'feature_importance': dict(zip(_self.feature_names, _self.xgb_model.feature_importances_))
            }
        
        # Store test data for later use
        results['test_data'] = {
            'X_test': X_test,
            'y_test': y_test,
            'X_test_scaled': X_test_scaled
        }
        
        logger.info("Model training completed successfully")
        return results
    
    @handle_error
    def save_models(self, models_dict: Dict[str, Any]):
        """Save trained models to disk"""
        try:
            # Save Random Forest
            if 'random_forest' in models_dict and models_dict['random_forest']['model']:
                rf_path = self.models_dir / "random_forest_model.joblib"
                joblib.dump(models_dict['random_forest']['model'], rf_path)
                logger.info(f"Random Forest model saved to {rf_path}")
            
            # Save XGBoost
            if 'xgboost' in models_dict and models_dict['xgboost']['model']:
                xgb_path = self.models_dir / "xgboost_model.joblib"
                joblib.dump(models_dict['xgboost']['model'], xgb_path)
                logger.info(f"XGBoost model saved to {xgb_path}")
            
            # Save scaler
            scaler_path = self.models_dir / "scaler.joblib"
            joblib.dump(self.scaler, scaler_path)
            
            # Save feature names
            features_path = self.models_dir / "feature_names.joblib"
            joblib.dump(self.feature_names, features_path)
            
            display_success("Models saved successfully!")
            
        except Exception as e:
            display_error(f"Error saving models: {str(e)}")
    
    @handle_error
    def load_models(self) -> bool:
        """Load saved models from disk"""
        try:
            # Load Random Forest
            rf_path = self.models_dir / "random_forest_model.joblib"
            if rf_path.exists():
                self.rf_model = joblib.load(rf_path)
                logger.info("Random Forest model loaded")
            
            # Load XGBoost
            xgb_path = self.models_dir / "xgboost_model.joblib"
            if xgb_path.exists() and XGBOOST_AVAILABLE:
                self.xgb_model = joblib.load(xgb_path)
                logger.info("XGBoost model loaded")
            
            # Load scaler
            scaler_path = self.models_dir / "scaler.joblib"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("Scaler loaded")
            
            # Load feature names
            features_path = self.models_dir / "feature_names.joblib"
            if features_path.exists():
                self.feature_names = joblib.load(features_path)
                logger.info("Feature names loaded")
            
            return True
            
        except Exception as e:
            display_error(f"Error loading models: {str(e)}")
            return False

class CustomerSegmentationModel:
    """Customer segmentation using K-Means clustering"""
    
    def __init__(self):
        self.kmeans_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    @st.cache_data(ttl=1800)  # Cache for 30 minutes
    def perform_segmentation(_self, df: pd.DataFrame, n_clusters: int = 5) -> Dict[str, Any]:
        """Perform customer segmentation"""
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")
        
        # Select features for clustering
        clustering_features = [
            'tenure',  'monthly_charges',
            'monthly_usage_gb', 'downtime_minutes', 'customer_satisfaction',
            'complaint_count', 'payment_delay_days', 'total_monthly_revenue'
        ]
        
        # Filter available features
        available_features = [col for col in clustering_features if col in df.columns]
        
        if len(available_features) < 3:
            raise ValueError("Insufficient features for clustering")
        
        # Prepare data
        X = df[available_features].copy()

        # --- FIX: Force convert all selected features to numeric ---
        for col in available_features:
            # Force conversion to numeric, invalid values become NaN
            # This handles mixed types, strings with currency symbols, etc.
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # --- IMPROVEMENT: Remove zero-variance features ---
        # Features with no variance (all values are the same) provide no information for clustering.
        # This is expanded to also remove very low variance features.
        variances = X.var()
        low_variance_cols = variances[variances < 1e-4].index
        if not low_variance_cols.empty:
            logger.info(f"Removing low-variance columns: {low_variance_cols.tolist()}")
            X = X.drop(columns=low_variance_cols)
        
        # Scale features
        X_scaled = _self.scaler.fit_transform(X)
        
        # Perform clustering
        _self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = _self.kmeans_model.fit_predict(X_scaled)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        
        # Add cluster labels to dataframe
        df_clustered = df.copy()
        df_clustered['cluster'] = cluster_labels
        
        # Analyze clusters
        cluster_summary = df_clustered.groupby('cluster').agg({
            'total_monthly_revenue': ['mean', 'sum', 'count'],
            'customer_satisfaction': 'mean',
            'churn': 'mean' if 'churn' in df_clustered.columns else lambda x: 0,
            'tenure': 'mean',
            'complaint_count': 'mean'
        }).round(2)
        
        results = {
            'clustered_data': df_clustered,
            'cluster_labels': cluster_labels,
            'cluster_summary': cluster_summary,
            'silhouette_score': silhouette_avg,
            'model': _self.kmeans_model,
            'scaler': _self.scaler,
            'feature_names': available_features
        }
        
        logger.info(f"Customer segmentation completed: {n_clusters} clusters, silhouette score: {silhouette_avg:.3f}")
        return results

class ExplainableAI:
    """Explainable AI using LIME and SHAP"""
    
    def __init__(self, model, X_train, feature_names):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.lime_explainer = None
        self.shap_explainer = None
        
        # Initialize LIME explainer if available
        if LIME_AVAILABLE:
            try:
                self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_train,
                    feature_names=feature_names,
                    class_names=['No Churn', 'Churn'],
                    mode='classification'
                )
            except Exception as e:
                logger.warning(f"Failed to initialize LIME explainer: {e}")
        
        # Initialize SHAP explainer if available
        if SHAP_AVAILABLE:
            try:
                self.shap_explainer = shap.TreeExplainer(model)
            except Exception as e:
                logger.warning(f"Failed to initialize SHAP explainer: {e}")
    
    @handle_error
    def explain_instance_lime(self, instance_idx: int, X_test: pd.DataFrame) -> Optional[Any]:
        """Explain single instance using LIME"""
        if not LIME_AVAILABLE or self.lime_explainer is None:
            display_warning("LIME not available. Please install: pip install lime")
            return None
        
        try:
            instance = X_test.iloc[instance_idx].values
            explanation = self.lime_explainer.explain_instance(
                instance, 
                self.model.predict_proba,
                num_features=10
            )
            return explanation
        except Exception as e:
            display_error(f"Error in LIME explanation: {str(e)}")
            return None
    
    @handle_error
    def explain_instance_shap(self, instance_idx: int, X_test: pd.DataFrame) -> Optional[Any]:
        """Explain single instance using SHAP"""
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            display_warning("SHAP not available. Please install: pip install shap")
            return None
        
        try:
            instance = X_test.iloc[[instance_idx]]
            shap_values = self.shap_explainer.shap_values(instance)
            return shap_values
        except Exception as e:
            display_error(f"Error in SHAP explanation: {str(e)}")
            return None
    
    @handle_error
    def create_global_shap_summary(self, X_test: pd.DataFrame, max_samples: int = 100) -> Optional[go.Figure]:
        """Create global SHAP summary plot"""
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            display_warning("SHAP not available. Please install: pip install shap")
            return None
        
        try:
            # Use a sample for performance
            sample_size = min(max_samples, len(X_test))
            X_sample = X_test.sample(n=sample_size, random_state=42)
            
            # Calculate SHAP values
            shap_values = self.shap_explainer.shap_values(X_sample)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # For binary classification, use the second element (class 1 - churn)
                if len(shap_values) > 1:
                    shap_array = np.array(shap_values[1])
                else:
                    shap_array = np.array(shap_values[0])
            else:
                shap_array = np.array(shap_values)
            
            # Ensure we have the right shape
            if shap_array.ndim == 3:
                shap_array = shap_array.reshape(shap_array.shape[0], -1)
            
            # Calculate mean absolute SHAP values for feature importance
            mean_abs_shap = np.mean(np.abs(shap_array), axis=0)
            
            # Create DataFrame for plotting
            feature_importance_df = pd.DataFrame({
                'Feature': self.feature_names,
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
                    text="<b>Global Feature Importance (SHAP)</b><br>Top Factors Driving Customer Churn",
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
            display_error(f"Error creating SHAP summary plot: {str(e)}")
            return None
    
    @handle_error
    def get_top_global_features(self, X_test: pd.DataFrame, top_n: int = 5) -> Dict[str, float]:
        """Get top global features with their importance scores"""
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            return {}
        
        try:
            # Use a sample for performance
            sample_size = min(100, len(X_test))
            X_sample = X_test.sample(n=sample_size, random_state=42)
            
            # Calculate SHAP values
            shap_values = self.shap_explainer.shap_values(X_sample)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                if len(shap_values) > 1:
                    shap_array = np.array(shap_values[1])
                else:
                    shap_array = np.array(shap_values[0])
            else:
                shap_array = np.array(shap_values)
            
            # Calculate mean absolute SHAP values
            if shap_array.ndim == 3:
                shap_array = shap_array.reshape(shap_array.shape[0], -1)
            
            mean_abs_shap = np.mean(np.abs(shap_array), axis=0)
            
            # Create feature importance dictionary
            feature_importance = dict(zip(self.feature_names, mean_abs_shap))
            
            # Get top N features
            top_features = dict(sorted(feature_importance.items(), 
                                     key=lambda x: x[1], reverse=True)[:top_n])
            
            return top_features
            
        except Exception as e:
            display_error(f"Error getting top global features: {str(e)}")
            return {}
    
    @handle_error
    def get_global_feature_importance(self, X_test: pd.DataFrame, max_samples: int = 100) -> Optional[Dict[str, float]]:
        """Get global feature importance using SHAP"""
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            return None
        
        try:
            # Use a sample for performance
            sample_size = min(max_samples, len(X_test))
            X_sample = X_test.sample(n=sample_size, random_state=42)
            
            shap_values = self.shap_explainer.shap_values(X_sample)
            # Normalize different possible SHAP outputs into a numeric array
            # shap_values can be:
            # - a single array (n_samples, n_features)
            # - a list/tuple of arrays (one per class) each (n_samples, n_features)
            # - a 3D array with classes/samples/features in various orders
            try:
                if isinstance(shap_values, (list, tuple)):
                    # stack along a new class axis if possible
                    try:
                        arr = np.stack([np.asarray(sv) for sv in shap_values], axis=0)
                    except Exception:
                        # fallback to simple conversion
                        arr = np.asarray(shap_values)
                else:
                    arr = np.asarray(shap_values)
            except Exception as e:
                display_error(f"Error converting SHAP values to array: {e}")
                return None

            n_features = len(self.feature_names)

            # Compute mean absolute SHAP importance per feature robustly
            try:
                if arr.ndim == 3:
                    # Try to detect which axis corresponds to features
                    if arr.shape[-1] == n_features:
                        # features are the last axis -> mean over other axes
                        mean_shap = np.mean(np.abs(arr), axis=tuple(range(arr.ndim - 1)))
                    elif arr.shape[1] == n_features:
                        # features are middle axis
                        mean_shap = np.mean(np.abs(arr), axis=(0, 2))
                    elif arr.shape[0] == n_features:
                        # features are first axis
                        mean_shap = np.mean(np.abs(arr), axis=(1, 2))
                    else:
                        # Flatten to (N, n_features) if possible
                        try:
                            arr2 = arr.reshape(-1, n_features)
                            mean_shap = np.mean(np.abs(arr2), axis=0)
                        except Exception:
                            raise ValueError(f"Unrecognized 3D SHAP shape: {arr.shape}")

                elif arr.ndim == 2:
                    # Most common: (n_samples, n_features)
                    if arr.shape[1] == n_features:
                        mean_shap = np.mean(np.abs(arr), axis=0)
                    elif arr.shape[0] == n_features and arr.shape[1] == 1:
                        mean_shap = np.mean(np.abs(arr), axis=1)
                    else:
                        # Try to coerce to (N, n_features)
                        try:
                            arr2 = arr.reshape(-1, n_features)
                            mean_shap = np.mean(np.abs(arr2), axis=0)
                        except Exception:
                            raise ValueError(f"Unrecognized 2D SHAP shape: {arr.shape}")

                elif arr.ndim == 1:
                    if arr.shape[0] == n_features:
                        mean_shap = np.abs(arr)
                    else:
                        raise ValueError(f"1D SHAP length {arr.shape[0]} != expected features {n_features}")

                else:
                    # fallback: attempt to ravel and reshape
                    arr2 = arr.ravel()
                    if arr2.size % n_features == 0:
                        arr2 = arr2.reshape(-1, n_features)
                        mean_shap = np.mean(np.abs(arr2), axis=0)
                    else:
                        raise ValueError(f"Unrecognized SHAP array shape: {arr.shape}")

            except Exception as e:
                display_error(f"Error computing mean SHAP values: {e}")
                return None

            # Create feature importance dictionary
            feature_importance = dict(zip(self.feature_names, mean_shap.tolist()))
            return feature_importance
            
        except Exception as e:
            display_error(f"Error calculating global feature importance: {str(e)}")
            return None