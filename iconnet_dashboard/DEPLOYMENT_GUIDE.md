# 🚀 ICONNET Dashboard - Deployment Guide

## 📋 Quick Start

### Option 1: Automated Installation
```bash
cd iconnet_dashboard
chmod +x install.sh
./install.sh
python3 run_dashboard.py
```

### Option 2: Manual Installation  
```bash
cd iconnet_dashboard
pip install -r requirements.txt
streamlit run main.py
```

## 🌐 Access Dashboard
- **Local URL**: http://localhost:8501
- **Network URL**: http://0.0.0.0:8501 (for remote access)

## 📂 Project Structure
```
iconnet_dashboard/
├── 📄 main.py                 # Main dashboard application
├── ⚙️  config.py              # Configuration settings
├── 🛠️  utils.py               # Utility functions
├── 📊 data_manager.py         # Data operations
├── 🤖 models.py               # ML models
├── 📈 visualizations.py       # Plotting functions
├── 📋 requirements.txt        # Dependencies
├── 🚀 run_dashboard.py        # Dashboard runner
├── 💾 install.sh              # Installation script
├── 📖 README.md               # Main documentation
├── 🔧 IMPROVEMENTS.md         # Code improvements details
├── 🚀 DEPLOYMENT_GUIDE.md     # This file
├── 📁 SourceData/             # CSV data files
├── 📁 models/                 # Saved ML models
└── 📁 logs/                   # Application logs
```

## 🎯 Dashboard Features

### 1. 📋 Data Overview
- **Sample Data Generation**: Generate realistic ICONNET customer data
- **File Upload**: Upload your own CSV files
- **Data Validation**: Comprehensive data quality checks  
- **Key Metrics**: Revenue, churn rate, customer satisfaction
- **Visualizations**: Revenue distribution, customer segments, contract types

### 2. 📉 Churn Analysis  
- **Churn Metrics**: Total churned customers, churn rate, lost revenue
- **Segment Analysis**: Churn rates by customer segment and service type
- **Factor Analysis**: Tenure, satisfaction, downtime impact on churn
- **Correlation Matrix**: Feature relationships visualization

### 3. 🔮 Predictive Modeling
- **Random Forest**: Robust ensemble method for churn prediction
- **XGBoost**: Advanced gradient boosting (if available)
- **Class Balancing**: Cluster-based undersampling for imbalanced data
- **Model Evaluation**: Confusion matrix, ROC curves, feature importance
- **Model Persistence**: Save/load trained models

### 4. 👥 Customer Segmentation
- **K-Means Clustering**: Automatic customer segmentation (3-8 clusters)
- **Silhouette Score**: Cluster quality evaluation
- **Cluster Analysis**: Revenue, satisfaction, churn by cluster
- **Detailed Views**: Individual cluster characteristics

### 5. 🔍 Explainable AI
- **LIME**: Local interpretable model explanations
- **SHAP**: SHapley additive explanations
- **Instance Explanations**: Why specific customers might churn
- **Global Importance**: Overall feature importance across all predictions

### 6. 💡 Strategic Recommendations
- **Revenue Optimization**: High-value customer retention strategies
- **Service Optimization**: Infrastructure and quality improvements
- **Segment Strategies**: Tailored approaches for each customer segment
- **Action Plans**: Immediate and medium-term recommendations
- **ROI Calculations**: Expected returns from interventions

### 7. ⚙️ Model Management
- **Model Persistence**: Save trained models to disk
- **Performance Reports**: Export model evaluation metrics
- **Data Export**: Download processed datasets
- **Version Control**: Track model versions and performance

## 🔧 Configuration

### Environment Variables
No environment variables required - dashboard is self-contained.

### Data Requirements
The dashboard expects CSV files with these columns:
- `customer_id`: Unique customer identifier
- `segment`: Customer segment (Residential, Corporate, Enterprise, Government)
- `tenure`: Months as customer
- `contract_duration`: Contract type (Monthly, 1 Year, 2 Years)
- `internet_speed_mbps`: Internet speed in Mbps
- `monthly_charges`: Base monthly charges
- `service_type`: Type of service
- `additional_services`: Additional services
- `monthly_usage_gb`: Monthly data usage in GB
- `downtime_minutes`: Monthly downtime in minutes
- `customer_satisfaction`: Satisfaction score (1-10)
- `payment_method`: Payment method
- `complaint_count`: Number of complaints
- `payment_delay_days`: Average payment delay
- `churn`: Churn status (0/1)
- `total_monthly_revenue`: Total monthly revenue

## 🚀 Production Deployment

### Docker Deployment (Recommended)
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
# Build and run
docker build -t iconnet-dashboard .
docker run -p 8501:8501 iconnet-dashboard
```

### Cloud Deployment Options

#### 1. Streamlit Cloud
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy with one click

#### 2. Heroku
```bash
# Create Procfile
echo "web: streamlit run main.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# Deploy
heroku create iconnet-dashboard
git push heroku main
```

#### 3. AWS EC2
```bash
# On EC2 instance
sudo apt update
sudo apt install python3-pip
git clone <your-repo>
cd iconnet_dashboard
pip3 install -r requirements.txt
nohup streamlit run main.py --server.port=8501 --server.address=0.0.0.0 &
```

### Load Balancing & Scaling
For high-traffic scenarios:
```yaml
# docker-compose.yml
version: '3.8'
services:
  dashboard:
    build: .
    ports:
      - "8501-8504:8501"
    deploy:
      replicas: 4
  
  nginx:
    image: nginx
    ports:
      - "80:80"
    depends_on:
      - dashboard
```

## 🔒 Security Considerations

### Authentication (Production)
```python
# Add to main.py for production
import streamlit_authenticator as stauth

authenticator = stauth.Authenticate(
    credentials,
    'cookie_name',
    'signature_key',
    cookie_expiry_days=30
)

name, authentication_status, username = authenticator.login('Login', 'main')
```

### Data Privacy
- Ensure CSV files don't contain PII
- Use environment variables for sensitive configs
- Implement data encryption for sensitive data

## 📊 Performance Optimization

### Caching Strategy
The dashboard uses Streamlit's caching:
```python
@st.cache_data(ttl=3600)  # Cache for 1 hour
def expensive_computation():
    pass
```

### Memory Management
- Large datasets automatically sampled
- Models cached to avoid retraining
- Efficient pandas operations

### Monitoring
```python
# Add to utils.py for monitoring
import psutil
import streamlit as st

def display_system_metrics():
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("CPU Usage", f"{cpu_percent}%")
    with col2:
        st.metric("Memory Usage", f"{memory_percent}%")
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Find process using port 8501
lsof -i :8501
# Kill process
kill -9 <PID>
```

#### 2. Missing Dependencies
```bash
# Reinstall all dependencies
pip install --force-reinstall -r requirements.txt
```

#### 3. LIME/SHAP Not Working
```bash
# Install optional dependencies
pip install lime shap
```

#### 4. Large File Upload Issues
```python
# Add to config.toml
[server]
maxUploadSize = 200
```

### Log Analysis
```bash
# View recent logs
tail -f logs/dashboard_$(date +%Y%m%d).log

# Search for errors
grep -i error logs/dashboard_*.log
```

## 📈 Monitoring & Analytics

### Dashboard Analytics
- Track user sessions
- Monitor model performance
- Log prediction accuracy

### Business Metrics
- Customer segmentation effectiveness
- Churn prediction accuracy
- Revenue optimization impact

## 🔄 Updates & Maintenance

### Regular Updates
1. **Data refresh**: Weekly data updates
2. **Model retraining**: Monthly model updates
3. **Performance monitoring**: Daily performance checks

### Backup Strategy
```bash
# Backup models and data
tar -czf backup_$(date +%Y%m%d).tar.gz SourceData/ models/ logs/
```

## 📞 Support

### Getting Help
1. Check `logs/` directory for error messages
2. Review `README.md` for detailed documentation
3. Check `IMPROVEMENTS.md` for recent changes

### Development
For development and customization:
```bash
# Development mode with auto-reload
streamlit run main.py --server.runOnSave=true
```

---

**Dashboard is ready for production deployment! 🚀**