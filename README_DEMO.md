# Pregnancy Risk Prediction System - Demo Application

A beautiful, interactive web application for predicting pregnancy risk using ensemble machine learning models (Random Forest + XGBoost).

## 🎯 Features

- **Interactive Input Forms** - Sliders and toggles for easy data entry
- **Real-Time Predictions** - Instant risk assessment using two ML models
- **Visual Analytics** - Gauge charts showing probability scores
- **Model Agreement Indicator** - Shows confidence through model convergence
- **Feature Importance** - Identifies top contributing risk factors
- **Clinical Recommendations** - Context-aware guidance based on risk level
- **Downloadable Reports** - Export detailed assessment reports
- **Professional UI** - Medical-themed color scheme and clean design

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Virtual environment (recommended)
- Trained model files (rf_model.pkl, xgb_model.pkl, scaler.pkl)

### Installation

1. **Navigate to your project directory:**
   ```bash
   cd "C:\Users\Quadir\Desktop\Final year project\development"
   ```

2. **Activate your virtual environment:**
   ```bash
   venv\Scripts\activate
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Ensure your models are in the correct location:**
   ```
   development/
   ├── app.py
   ├── requirements.txt
   ├── models/
   │   ├── rf_model.pkl
   │   ├── xgb_model.pkl
   │   └── scaler.pkl
   ```

2. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

3. **Access the application:**
   - The app will automatically open in your default browser
   - Default URL: http://localhost:8501
   - If it doesn't open automatically, navigate to the URL shown in terminal

## 📖 Usage Guide

### Step 1: Enter Patient Information

**Vital Signs & Measurements:**
- Age (18-50 years)
- Systolic Blood Pressure (80-200 mmHg)
- Diastolic Blood Pressure (50-130 mmHg)
- Blood Sugar (3.0-20.0 mmol/L)
- Body Temperature (95-105 °F)

**Physical & Metabolic:**
- Body Mass Index (15-50)
- Heart Rate (50-150 bpm)

**Medical History (checkboxes):**
- Previous Pregnancy Complications
- Preexisting Diabetes
- Gestational Diabetes
- Mental Health Concerns

### Step 2: Assess Risk

Click the **"🔍 Assess Pregnancy Risk"** button to generate predictions.

### Step 3: Review Results

The system displays:

1. **Risk Classification Banner**
   - High Risk (red) or Low Risk (green)
   - Ensemble confidence percentage

2. **Model Predictions**
   - Three gauge charts (Random Forest, XGBoost, Ensemble)
   - Individual probability scores
   - Visual risk indicators

3. **Model Agreement**
   - Agreement status (Strong/Moderate/Low)
   - Probability difference between models
   - Confidence indicator

4. **Contributing Factors**
   - Top 5 features influencing the prediction
   - Feature importance percentages
   - Actual patient values

5. **Feature Importance Chart**
   - Horizontal bar chart
   - Visual ranking of factors

6. **Clinical Recommendations**
   - Context-aware guidance
   - Action items based on risk level
   - Follow-up suggestions

### Step 4: Generate Report (Optional)

Click **"📄 Generate Detailed Report"** to download a text file containing:
- Complete patient information
- All prediction scores
- Contributing factors
- Clinical recommendations

## 🎨 UI Features

### Color Scheme
- **Blue tones** - Primary interface elements
- **Green** - Low risk indicators
- **Red** - High risk warnings
- **Orange** - Moderate risk/agreement warnings

### Interactive Elements
- **Sliders** - Continuous numerical inputs
- **Number inputs** - Precise value entry
- **Checkboxes** - Binary medical history
- **Gauge charts** - Intuitive probability visualization
- **Bar charts** - Feature importance comparison

## 🔧 Customization

### Changing Model Paths

If your models are in a different location, edit `app.py`:

```python
@st.cache_resource
def load_models():
    models_path = Path("your/custom/path")  # Change this line
    ...
```

### Adjusting Risk Threshold

The default classification threshold is 50%. To change:

```python
# In the prediction section
risk_level = "High Risk" if ensemble_prob >= 0.5 else "Low Risk"  # Change 0.5
```

### Modifying Color Scheme

Edit the custom CSS in the `st.markdown()` section at the top of `app.py`.

## 📊 Model Information

### Performance Metrics
- **Accuracy:** 99.43%
- **Precision:** 98.59%
- **Recall:** 100% (perfect - no high-risk cases missed)
- **F1-Score:** 99.29%
- **ROC-AUC:** 0.9995+

### Training Data
- **Samples:** 1,170 maternal health records
- **Features:** 11 clinical and demographic variables
- **Class Distribution:** 60% Low Risk, 40% High Risk
- **Algorithms:** Random Forest (100 trees) + XGBoost

### Feature Importance (Top 5)
1. Preexisting Diabetes (19.86%)
2. Blood Sugar (18.99%)
3. BMI (18.16%)
4. Heart Rate (13.11%)
5. Mental Health (12.17%)

## 🐛 Troubleshooting

### "Model Loading Error"

**Problem:** Models not found

**Solution:**
- Check that model files exist in `models/` directory
- Verify filenames match exactly: `rf_model.pkl`, `xgb_model.pkl`, `scaler.pkl`
- Check file paths in the error message

### "ModuleNotFoundError"

**Problem:** Missing package

**Solution:**
```bash
pip install -r requirements.txt
```

### Port Already in Use

**Problem:** Port 8501 is busy

**Solution:**
```bash
streamlit run app.py --server.port 8502
```

### App Won't Open in Browser

**Problem:** Auto-launch failed

**Solution:**
- Manually navigate to http://localhost:8501
- Check terminal for the correct URL
- Try: `streamlit run app.py --server.headless false`

## 📱 Deployment Options

### Local Network Access

Share with colleagues on same network:
```bash
streamlit run app.py --server.address 0.0.0.0
```

Access from other devices: `http://YOUR_IP:8501`

### Cloud Deployment (Future)

The app can be deployed to:
- **Streamlit Cloud** (free, easiest)
- **Heroku** (requires Procfile)
- **AWS/Azure** (for production)

See Streamlit documentation for deployment guides.

## 🔒 Important Notes

### Clinical Use Disclaimer

This system provides **decision support**, not medical diagnosis:
- Always use clinical judgment
- Consider additional patient context
- Verify predictions with standard protocols
- System augments, does not replace, healthcare providers

### Data Privacy

- No patient data is stored by the application
- All processing happens locally
- No data is sent to external servers
- Each session is independent

### Limitations

- Requires complete data (no missing values)
- Performance may vary on different populations
- Based on specific training dataset
- Should undergo validation before clinical deployment

## 📚 Technical Details

### Architecture

```
User Input → Preprocessing → Models → Ensemble → Display
                ↓
         Feature Scaling
         (StandardScaler)
                ↓
    Random Forest ← → XGBoost
                ↓
         Average Probabilities
                ↓
         Risk Classification
```

### File Structure

```
development/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── models/
│   ├── rf_model.pkl         # Trained Random Forest
│   ├── xgb_model.pkl        # Trained XGBoost
│   └── scaler.pkl           # StandardScaler
├── src/                     # Training scripts
├── notebooks/               # Analysis notebooks
├── data/                    # Dataset
└── results/                 # Figures and metrics
```

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review Streamlit documentation: https://docs.streamlit.io
3. Check model training logs for compatibility

## 🙏 Acknowledgments

- Developed as part of undergraduate research
- Federal University of Agriculture, Abeokuta
- Machine Learning for Maternal Healthcare

## 📄 License

This is academic research software. Use for educational purposes.

---

**Version:** 1.0.0  
**Last Updated:** March 2026  
**Author:** Muhammad Abdulquadir Akanfe