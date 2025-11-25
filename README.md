# 📊 ICONNET Predictive Analytics Dashboard

## 🎯 Overview
Dashboard analisis prediktif untuk optimalisasi pendapatan layanan ICONNET di PT PLN ICON PLUS SBU Regional Jawa Tengah. Dashboard ini menggunakan machine learning untuk prediksi churn dan segmentasi pelanggan.

## ✨ Features

### 1. 📋 Data Overview
- Visualisasi data pelanggan
- Statistik pendapatan dan churn
- Distribusi segmen pelanggan

### 2. 📉 Churn Analysis
- Analisis tingkat churn per segmen
- Faktor-faktor penyebab churn
- Korelasi antar variabel

### 3. 🔮 Predictive Modeling
- **Random Forest Classifier**
- **XGBoost Classifier**
- Handling class imbalance
- ROC curves dan feature importance

### 4. 👥 Customer Segmentation
- **K-Means Clustering**
- Analisis optimal cluster
- Profil segmen pelanggan

### 5. 🤖 Explainable AI
- **LIME** explanations
- **SHAP** values
- Model interpretability

### 6. 📈 Strategic Recommendations
- Rekomendasi bisnis
- ROI analysis
- Implementation roadmap

### 7. 🔧 Model Management
- Save/load trained models
- Model information

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation & Running

1. **Install dependencies:**
   ```bash
   pip install -r requirements_iconnet.txt
   ```

2. **Run the dashboard:**
   ```bash
   streamlit run iconnet_dashboard/main.py.py
   ```

3. **Or use the automated script:**
   ```bash
   chmod +x iconnet_dashboard/run_dashboard.sh
   ./iconnet_dashboard/run_dashboard.sh
   ```

4. **Access the dashboard:**
   Open your browser and go to `http://localhost:8501`


## 🔧 Technical Details

### Libraries Used
- **Streamlit**: Web dashboard framework
- **Pandas & NumPy**: Data manipulation
- **Scikit-learn**: Machine learning algorithms
- **XGBoost**: Gradient boosting
- **Plotly**: Interactive visualizations
- **LIME & SHAP**: Model interpretability
- **Matplotlib & Seaborn**: Statistical plots

### Machine Learning Models
1. **Random Forest**: Ensemble method with balanced classes
2. **XGBoost**: Gradient boosting with scale_pos_weight
3. **K-Means**: Customer segmentation clustering

### Data Features
- Customer demographics and behavior
- Service usage patterns
- Payment and contract information
- Service quality metrics
- Customer satisfaction scores

## 🎨 UI Features

### Responsive Design
- Mobile-friendly layout
- Interactive visualizations
- Professional styling

### Navigation
- Sidebar navigation
- Tab-based sections
- Expandable information panels

### Error Handling
- Missing file fallbacks
- Graceful error recovery
- User-friendly error messages

## 📊 Business Impact

### Key Metrics
- **Churn Rate**: Monitor customer retention
- **Revenue Analysis**: Track monthly revenue patterns
- **Segmentation**: Identify high-value customers
- **Satisfaction**: Monitor service quality

### Strategic Benefits
- **15% churn reduction potential**
- **20% revenue growth opportunities**
- **Improved customer experience**
- **Data-driven decision making**

## 🔍 Usage Guide

### For Business Users
1. Start with **Data Overview** for general insights
2. Check **Churn Analysis** for retention metrics
3. Review **Strategic Recommendations** for action items

### For Technical Users
1. Explore **Predictive Modeling** for model performance
2. Use **Explainable AI** for model interpretability
3. Utilize **Model Management** for saving/loading models

### For Data Scientists
1. Analyze **Customer Segmentation** for clustering insights
2. Examine feature importance and SHAP values
3. Customize models and parameters as needed

## 🚨 Troubleshooting

### Common Issues
1. **Logo files missing**: Dashboard will use fallback designs
2. **LIME/SHAP errors**: Check library versions and compatibility
3. **Memory issues**: Reduce sample sizes for large datasets

### Performance Tips
- Use data sampling for large datasets
- Clear browser cache if visualizations don't load
- Restart the dashboard if memory usage is high

## 👥 Team
- **Hani Setiawan** (hani.setiawan@binus.ac.id)
- **Jetbar Runggu Hamonangan Doloksaribu** (jetbar.doloksaribu@binus.ac.id)
- **Naufal Yafi** (naufal.yafi@binus.ac.id)

## 📧 Support
Untuk pertanyaan teknis atau dukungan, silakan hubungi tim pengembang.

---
© 2025 PT PLN ICON PLUS SBU Regional Jawa Tengah