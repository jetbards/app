# 📊 ICONNET Predictive Analytics Dashboard

## 🌟 Overview

Dashboard analitik prediktif yang komprehensif untuk mengoptimalkan pendapatan layanan ICONNET di PT PLN ICON PLUS SBU Regional Jawa Tengah. Dashboard ini menggunakan machine learning untuk prediksi churn, segmentasi pelanggan, dan memberikan rekomendasi strategis berbasis data.

## ✨ Fitur Utama

### 📋 Data Overview
- Visualisasi distribusi data pelanggan
- Metrics kinerja utama (KPI)
- Analisis segmen pelanggan
- Distribusi revenue dan kontrak

### 📉 Churn Analysis
- Analisis tingkat churn per segmen
- Identifikasi faktor-faktor churn
- Visualisasi korelasi fitur
- Box plots untuk analisis mendalam

### 🔮 Predictive Modeling
- **Random Forest**: Model ensemble untuk prediksi churn yang robust
- **XGBoost**: Gradient boosting dengan optimasi lanjutan
- **Class Balancing**: Penanganan data tidak seimbang dengan cluster-based undersampling
- ROC curves dan Precision-Recall curves
- Feature importance analysis

### 👥 Customer Segmentation
- **K-Means Clustering**: Segmentasi pelanggan otomatis
- Silhouette score untuk evaluasi kualitas cluster
- Analisis karakteristik per cluster
- Visualisasi multidimensional

### 🔍 Explainable AI
- **LIME**: Local Interpretable Model-agnostic Explanations
- **SHAP**: SHapley Additive exPlanations
- Penjelasan prediksi level instance dan global
- Visualisasi feature importance yang interaktif

### 💡 Strategic Recommendations
- Rekomendasi berbasis data untuk optimasi revenue
- Strategi retensi pelanggan high-value
- Action plan jangka pendek dan menengah
- Perhitungan ROI potensial

### ⚙️ Model Management
- Save/load model ke disk
- Export laporan performa model
- Versioning dan tracking model
- Export data untuk analisis lanjutan

## 🏗️ Arsitektur Modular

Dashboard ini dibangun dengan arsitektur modular yang clean dan maintainable:

```
iconnet_dashboard/
├── main.py                 # Dashboard utama
├── config.py              # Konfigurasi dan konstanta
├── utils.py               # Utility functions dan error handling
├── data_manager.py        # Data loading dan validasi
├── models.py              # Machine learning models
├── visualizations.py      # Plotly visualizations
├── requirements.txt       # Dependencies
├── run_dashboard.py       # Script untuk menjalankan dashboard
├── README.md             # Dokumentasi ini
├── SourceData/           # Folder untuk data CSV
├── models/               # Folder untuk menyimpan trained models
└── logs/                 # Folder untuk log files
```

## 🚀 Instalasi dan Penggunaan

### Prerequisites
- Python 3.7+
- pip atau conda

### Instalasi

1. **Clone atau download folder dashboard:**
   ```bash
   cd iconnet_dashboard
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan dashboard:**
   ```bash
   # Menggunakan script runner
   python run_dashboard.py
   
   # Atau langsung dengan streamlit
   streamlit run main.py
   ```

4. **Buka browser:**
   Dashboard akan tersedia di `http://localhost:8501`

### Penggunaan Data

#### Option 1: Generate Sample Data
- Pilih "Generate Sample Data" di sidebar
- Atur jumlah customer (100-5000)
- Klik "Generate Data"

#### Option 2: Upload CSV File
- Pilih "Upload CSV File" di sidebar
- Upload file CSV dengan struktur yang sesuai
- Data akan disimpan di folder `SourceData/`

#### Option 3: Load Existing File
- Pilih "Load Existing File" di sidebar
- Pilih dari file CSV yang tersedia di `SourceData/`

## 📊 Format Data

Dashboard mengharapkan data dengan kolom-kolom berikut:

| Kolom | Deskripsi | Tipe |
|-------|-----------|------|
| customer_id | ID unik pelanggan | String |
| segment | Segmen pelanggan (Residential, Corporate, Enterprise, Government) | String |
| tenure | Lama berlangganan (bulan) | Integer |
| contract_duration | Durasi kontrak (Monthly, 1 Year, 2 Years) | String |
| internet_speed_mbps | Kecepatan internet (Mbps) | Float |
| monthly_charges | Biaya bulanan dasar | Float |
| service_type | Jenis layanan | String |
| additional_services | Layanan tambahan | String |
| monthly_usage_gb | Penggunaan bulanan (GB) | Float |
| downtime_minutes | Downtime per bulan (menit) | Float |
| customer_satisfaction | Kepuasan pelanggan (1-10) | Integer |
| payment_method | Metode pembayaran | String |
| complaint_count | Jumlah komplain | Integer |
| payment_delay_days | Keterlambatan pembayaran (hari) | Float |
| churn | Status churn (0/1) | Integer |
| total_monthly_revenue | Total revenue bulanan | Float |

## 🔧 Perbaikan dari Versi Sebelumnya

### ✅ Code Quality Improvements
- **Modular Architecture**: Code dipecah menjadi modul-modul terpisah
- **Error Handling**: Robust exception handling dengan logging
- **Caching**: Streamlit caching untuk performa optimal
- **Type Hints**: Type annotations untuk better code documentation
- **Clean Code**: Consistent naming dan dokumentasi

### ✅ Performance Optimization
- **@st.cache_data**: Cache untuk operasi yang expensive
- **Lazy Loading**: Load components hanya saat dibutuhkan
- **Efficient Data Processing**: Optimasi operasi pandas dan sklearn
- **Memory Management**: Better handling untuk large datasets

### ✅ Enhanced Features
- **Progress Tracking**: Progress bars untuk operasi yang lama
- **Model Persistence**: Save/load trained models
- **Data Validation**: Comprehensive data validation
- **Export Functionality**: Export data dan reports
- **Logging**: Structured logging untuk debugging

### ✅ UI/UX Improvements
- **Consistent Styling**: Custom CSS dengan theme yang konsisten
- **Better Navigation**: Intuitive sidebar navigation
- **Responsive Design**: Adaptif untuk berbagai ukuran screen
- **Error Messages**: User-friendly error messages
- **Help Text**: Contextual help dan tooltips

## 🎯 Use Cases

### 1. Churn Prediction
- Identifikasi pelanggan berisiko tinggi untuk churn
- Implementasi campaign retensi yang targeted
- Monitor effectiveness dari retention strategies

### 2. Customer Segmentation
- Grouping pelanggan berdasarkan behavior dan value
- Personalisasi offering per segment
- Optimasi resource allocation

### 3. Revenue Optimization
- Identifikasi opportunity untuk upselling
- Optimasi pricing strategy
- Fokus pada high-value customer retention

### 4. Operational Efficiency
- Identifikasi area improvement dalam service quality
- Prioritasi infrastructure investments
- Monitor customer satisfaction metrics

## 🔬 Technical Details

### Machine Learning Models

#### Random Forest
- **Parameters**: 100 trees, max depth 10, balanced class weights
- **Advantages**: Robust, handles mixed data types, feature importance
- **Use Case**: Primary churn prediction model

#### XGBoost
- **Parameters**: 100 trees, learning rate 0.1, max depth 6
- **Advantages**: High performance, gradient boosting
- **Use Case**: Comparison model untuk validation

#### K-Means Clustering
- **Parameters**: 5 clusters default, configurable
- **Evaluation**: Silhouette score
- **Use Case**: Customer segmentation

### Data Processing
- **Missing Values**: Median imputation untuk numerical features
- **Categorical Encoding**: One-hot encoding
- **Feature Scaling**: StandardScaler untuk clustering
- **Class Imbalance**: Cluster-based undersampling

### Explainable AI
- **LIME**: Local explanations untuk individual predictions
- **SHAP**: Global dan local feature importance
- **Visualization**: Interactive charts untuk interpretability

## 🚀 Future Enhancements

### Planned Features
- [ ] **Real-time Data Integration**: Connect ke database production
- [ ] **Advanced Models**: Deep learning models
- [ ] **A/B Testing**: Framework untuk testing interventions
- [ ] **API Integration**: REST API untuk model serving
- [ ] **Automated Reporting**: Scheduled reports via email
- [ ] **Mobile Responsive**: Better mobile experience

### Technical Improvements
- [ ] **Unit Tests**: Comprehensive test coverage
- [ ] **CI/CD Pipeline**: Automated testing dan deployment
- [ ] **Docker Support**: Containerization
- [ ] **Database Integration**: PostgreSQL/MongoDB support
- [ ] **Authentication**: User management system

## 🤝 Contributing

Untuk kontribusi pada dashboard ini:

1. Fork repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

Project ini dibuat untuk PT PLN ICON PLUS SBU Regional Jawa Tengah sebagai bagian dari optimalisasi pendapatan layanan ICONNET.

## 👥 Tim Pengembang

- **Hani Setiawan** (2702464202)
- **Jetbar Runggu Hamonangan Doloksaribu** (2702462973)  
- **Naufal Yafi** (2702476240)

**Institusi**: BINUS University  
**Client**: PT PLN ICON PLUS SBU Regional Jawa Tengah

## 📞 Support

Untuk pertanyaan atau support:
- Check logs di folder `logs/`
- Review dokumentasi di README ini
- Contact tim pengembang

---

**Happy Analytics! 📊✨**