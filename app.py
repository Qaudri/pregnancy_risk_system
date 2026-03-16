"""
Pregnancy Risk Prediction System - Streamlit Demo
Machine Learning-Based Binary Risk Classification for Antenatal Care

This application provides an interactive interface for predicting pregnancy risk
using ensemble machine learning models (Random Forest + XGBoost).
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

try:
    import plotly.graph_objects as go  # type: ignore[reportMissingImports]
    PLOTLY_AVAILABLE = True
except ModuleNotFoundError:
    go = None
    PLOTLY_AVAILABLE = False

NUMERICAL_FEATURES = [
    'Age',
    'Systolic BP',
    'Diastolic',
    'BS',
    'Body Temp',
    'BMI',
    'Heart Rate'
]

BOOLEAN_FEATURES = [
    'Previous Complications',
    'Preexisting Diabetes',
    'Gestational Diabetes',
    'Mental Health'
]

ALL_FEATURES = NUMERICAL_FEATURES + BOOLEAN_FEATURES

# Page configuration
st.set_page_config(
    page_title="Pregnancy Risk Prediction System",
    page_icon="🤰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: rgba(127, 29, 29, 0.88);
        border-left: 5px solid #ff4444;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #f8fafc;
    }
    .risk-low {
        background-color: rgba(20, 83, 45, 0.88);
        border-left: 5px solid #44ff44;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #f8fafc;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .risk-high h2, .risk-high p, .risk-high strong,
    .risk-low h2, .risk-low p, .risk-low strong {
        color: #f8fafc !important;
    }
    .feature-contribution {
        background-color: rgba(15, 23, 42, 0.82);
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.3rem 0;
        border-left: 3px solid #1f77b4;
        border: 1px solid rgba(148, 163, 184, 0.25);
        color: #f8fafc;
    }
    .feature-contribution strong {
        color: #f8fafc;
    }
    .feature-meta {
        color: #cbd5e1;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-size: 1.1rem;
        padding: 0.5rem 2rem;
        border-radius: 10px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #145a8c;
    }
    .stDownloadButton>button {
        background-color: #1f77b4;
        color: white;
        font-size: 1.05rem;
        padding: 0.5rem 1.25rem;
        border-radius: 10px;
        border: none;
        width: fit-content;
    }
    .stDownloadButton>button:hover {
        background-color: #145a8c;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load trained models and scaler"""
    try:
        # Adjust paths based on your project structure
        models_path = Path("models")
        
        rf_model = joblib.load(models_path / "rf_model.pkl")
        xgb_model = joblib.load(models_path / "xgb_model.pkl")
        scaler = joblib.load(models_path / "scaler.pkl")
        
        return rf_model, xgb_model, scaler, None
    except Exception as e:
        return None, None, None, str(e)


def create_gauge_chart(probability, model_name):
    """Create a gauge chart for risk probability"""
    if not PLOTLY_AVAILABLE:
        return None
    
    # Determine color based on probability
    if probability < 0.3:
        color = "green"
    elif probability < 0.7:
        color = "orange"
    else:
        color = "red"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{model_name}", 'font': {'size': 20, 'color': '#e5e7eb'}},
        number={'suffix': "%", 'font': {'size': 40, 'color': '#f8fafc'}},
        gauge={
            'axis': {
                'range': [None, 100],
                'tickwidth': 1,
                'tickcolor': "#cbd5e1",
                'tickfont': {'color': '#cbd5e1'}
            },
            'bar': {'color': color},
            'bgcolor': "rgba(15, 23, 42, 0.0)",
            'borderwidth': 2,
            'bordercolor': "rgba(148, 163, 184, 0.35)",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(34, 197, 94, 0.18)'},
                {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.18)'},
                {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.18)'}
            ],
            'threshold': {
                'line': {'color': "#e5e7eb", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#e5e7eb", 'family': "Arial"}
    )
    
    return fig


def create_feature_importance_chart(feature_names, feature_values, shap_values=None):
    """Create horizontal bar chart for feature contributions"""
    if not PLOTLY_AVAILABLE:
        return None
    
    # If SHAP values provided, use them; otherwise use feature values
    if shap_values is not None:
        contributions = shap_values
        title = "SHAP Feature Contributions"
    else:
        # Simple contribution estimate based on deviation from mean
        contributions = np.abs(feature_values - 0.5)  # Assuming scaled features
        title = "Feature Importance (Approximate)"
    
    # Get top 5 features
    top_indices = np.argsort(np.abs(contributions))[-5:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    top_contributions = [contributions[i] for i in top_indices]
    
    colors = ['#ff6b6b' if c > 0 else '#51cf66' for c in top_contributions]
    
    fig = go.Figure(go.Bar(
        x=top_contributions,
        y=top_features,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(0,0,0,0.3)', width=1)
        ),
        text=[f"{abs(c):.3f}" for c in top_contributions],
        textposition='auto',
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Contribution",
        yaxis_title="Feature",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12, color='#e5e7eb'),
        title_font=dict(color='#f8fafc')
    )
    
    fig.update_traces(textfont=dict(color='#f8fafc', size=12))
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(148, 163, 184, 0.35)',
        tickfont=dict(color='#cbd5e1'),
        title_font=dict(color='#e5e7eb')
    )
    fig.update_yaxes(
        tickfont=dict(color='#cbd5e1'),
        title_font=dict(color='#e5e7eb')
    )
    
    return fig


def render_probability_visual(probability, model_name):
    """Render a model probability using Plotly when available, with a Streamlit fallback."""
    if PLOTLY_AVAILABLE:
        st.plotly_chart(
            create_gauge_chart(probability, model_name),
            use_container_width=True
        )
        return

    st.metric(f"{model_name} Probability", f"{probability * 100:.1f}%")
    st.progress(int(round(probability * 100)))


def render_feature_importance_visual(feature_names, feature_values, feature_importance):
    """Render feature importance with a Plotly chart or a simple table fallback."""
    if PLOTLY_AVAILABLE:
        st.plotly_chart(
            create_feature_importance_chart(
                feature_names,
                feature_values,
                feature_importance
            ),
            use_container_width=True
        )
        return

    top_indices = np.argsort(feature_importance)[-5:][::-1]
    fallback_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in top_indices],
        'Scaled Value': [round(float(feature_values[i]), 3) for i in top_indices],
        'Importance (%)': [round(float(feature_importance[i]) * 100, 1) for i in top_indices]
    })
    st.table(fallback_df)


def get_clinical_recommendation(risk_level, probability):
    """Generate clinical recommendation based on risk level"""
    
    if risk_level == "High Risk":
        if probability > 0.9:
            return """
            **Immediate Actions Required:**
            - Schedule urgent specialist consultation
            - Enhanced antenatal monitoring required
            - Weekly follow-up appointments recommended
            - Consider referral to tertiary care facility
            - Comprehensive risk factor management plan needed
            """
        else:
            return """
            **Enhanced Monitoring Recommended:**
            - Specialist consultation within 2 weeks
            - Increased frequency of antenatal visits
            - Close monitoring of identified risk factors
            - Consider additional diagnostic tests
            - Patient education on warning signs
            """
    else:
        return """
        **Standard Care Appropriate:**
        - Continue routine antenatal care schedule
        - Regular monitoring of vital signs
        - Maintain healthy lifestyle practices
        - Report any unusual symptoms promptly
        - Next scheduled visit as per standard protocol
        """


def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤰 Pregnancy Risk Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Machine Learning-Based Binary Risk Classification for Antenatal Care</p>', unsafe_allow_html=True)
    
    # Load models
    rf_model, xgb_model, scaler, error = load_models()
    
    if error:
        st.error(f"""
        **Model Loading Error:** {error}
        
        Please ensure the following files exist in the `models/` directory:
        - rf_model.pkl
        - xgb_model.pkl
        - scaler.pkl
        
        Current working directory: {Path.cwd()}
        """)
        st.stop()
    
    st.success("✅ Models loaded successfully!")
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly is not installed. Showing simplified visual summaries instead of interactive charts.")
    
    # Sidebar for instructions
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. **Enter patient information** in the form below
        2. **Adjust all parameters** using sliders and toggles
        3. **Click 'Assess Risk'** to generate prediction
        4. **Review results** including:
           - Risk classification
           - Probability scores
           - Contributing factors
           - Clinical recommendations
        
        ---
        
        ### About the System
        
        This system uses ensemble machine learning:
        - **Random Forest** (100 decision trees)
        - **XGBoost** (Gradient Boosting)
        
        **Performance:**
        - Accuracy: 99.43%
        - Recall: 100%
        - Precision: 98.59%
        
        **Models trained on:** 1,170 maternal health records
        """)
        
        st.markdown("---")
        st.caption("Developed as part of undergraduate research in AI for Healthcare")
    
    # Main content area
    st.header("Patient Information")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Vital Signs & Measurements")
        
        age = st.slider(
            "Age (years)",
            min_value=18,
            max_value=50,
            value=28,
            help="Maternal age at time of assessment"
        )
        
        systolic_bp = st.number_input(
            "Systolic Blood Pressure (mmHg)",
            min_value=80,
            max_value=200,
            value=120,
            step=1,
            help="Upper blood pressure reading"
        )
        
        diastolic_bp = st.number_input(
            "Diastolic Blood Pressure (mmHg)",
            min_value=50,
            max_value=130,
            value=80,
            step=1,
            help="Lower blood pressure reading"
        )
        
        blood_sugar = st.number_input(
            "Blood Sugar (mmol/L)",
            min_value=3.0,
            max_value=20.0,
            value=7.0,
            step=0.1,
            help="Blood glucose level"
        )
        
        body_temp = st.number_input(
            "Body Temperature (°F)",
            min_value=95.0,
            max_value=105.0,
            value=98.6,
            step=0.1,
            help="Body temperature in Fahrenheit"
        )
    
    with col2:
        st.subheader("📈 Physical & Metabolic")
        
        bmi = st.number_input(
            "Body Mass Index (BMI)",
            min_value=15.0,
            max_value=50.0,
            value=24.0,
            step=0.1,
            help="Weight (kg) / Height (m)²"
        )
        
        heart_rate = st.number_input(
            "Heart Rate (bpm)",
            min_value=50,
            max_value=150,
            value=75,
            step=1,
            help="Beats per minute"
        )
        
        st.subheader("🏥 Medical History")
        
        previous_complications = st.checkbox(
            "Previous Pregnancy Complications",
            help="Any complications in previous pregnancies"
        )
        
        preexisting_diabetes = st.checkbox(
            "Preexisting Diabetes",
            help="Type 1 or Type 2 diabetes before pregnancy"
        )
        
        gestational_diabetes = st.checkbox(
            "Gestational Diabetes",
            help="Diabetes diagnosed during current pregnancy"
        )
        
        mental_health = st.checkbox(
            "Mental Health Concerns",
            help="Depression, anxiety, or other mental health issues"
        )
    
    st.markdown("---")
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔍 Assess Pregnancy Risk", use_container_width=True)
    
    if predict_button:
        # Prepare input data
        feature_names = ALL_FEATURES
        
        input_data = pd.DataFrame({
            'Age': [age],
            'Systolic BP': [systolic_bp],
            'Diastolic': [diastolic_bp],
            'BS': [blood_sugar],
            'Body Temp': [body_temp],
            'BMI': [bmi],
            'Heart Rate': [heart_rate],
            'Previous Complications': [1 if previous_complications else 0],
            'Preexisting Diabetes': [1 if preexisting_diabetes else 0],
            'Gestational Diabetes': [1 if gestational_diabetes else 0],
            'Mental Health': [1 if mental_health else 0]
        })[feature_names]
        
        # Scale only the numerical features used during preprocessing
        input_prepared = input_data.copy()
        input_prepared[NUMERICAL_FEATURES] = scaler.transform(input_data[NUMERICAL_FEATURES])
        
        # Make predictions
        rf_prob = rf_model.predict_proba(input_prepared)[0][1]
        xgb_prob = xgb_model.predict_proba(input_prepared)[0][1]
        ensemble_prob = (rf_prob + xgb_prob) / 2
        
        # Determine risk level
        risk_level = "High Risk" if ensemble_prob >= 0.5 else "Low Risk"
        
        # Display results
        st.markdown("---")
        st.header("🎯 Risk Assessment Results")
        
        # Risk classification banner
        if risk_level == "High Risk":
            st.markdown(f"""
            <div class="risk-high">
                <h2 style="color: #cc0000; margin: 0;">⚠️ HIGH RISK PREGNANCY DETECTED</h2>
                <p style="font-size: 1.2rem; margin-top: 0.5rem;">
                    Ensemble Confidence: <strong>{ensemble_prob*100:.1f}%</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-low">
                <h2 style="color: #00aa00; margin: 0;">✅ LOW RISK PREGNANCY</h2>
                <p style="font-size: 1.2rem; margin-top: 0.5rem;">
                    Ensemble Confidence: <strong>{(1-ensemble_prob)*100:.1f}%</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Model predictions visualization
        st.subheader("📊 Model Predictions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            render_probability_visual(rf_prob, "Random Forest")
            if rf_prob >= 0.5:
                st.markdown("**Prediction:** High Risk")
            else:
                st.markdown("**Prediction:** Low Risk")
        
        with col2:
            render_probability_visual(xgb_prob, "XGBoost")
            if xgb_prob >= 0.5:
                st.markdown("**Prediction:** High Risk")
            else:
                st.markdown("**Prediction:** Low Risk")
        
        with col3:
            render_probability_visual(ensemble_prob, "Ensemble")
            st.markdown(f"**Final Prediction:** {risk_level}")
        
        # Model Agreement
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            # Agreement indicator
            agreement = abs(rf_prob - xgb_prob)
            if agreement < 0.1:
                agreement_status = "🟢 Strong Agreement"
                agreement_color = "green"
            elif agreement < 0.2:
                agreement_status = "🟡 Moderate Agreement"
                agreement_color = "orange"
            else:
                agreement_status = "🔴 Low Agreement"
                agreement_color = "red"
            
            st.markdown(f"""
            ### Model Agreement
            **Status:** <span style="color: {agreement_color};">{agreement_status}</span>
            
            **Probability Difference:** {agreement*100:.2f}%
            
            Both models showing {'strong' if agreement < 0.1 else 'some'} convergence on the prediction.
            """, unsafe_allow_html=True)
        
        with col2:
            # Top contributing factors
            st.markdown("### 🔍 Top Contributing Factors")
            
            # Get feature importances (simplified - using Random Forest)
            feature_importance = rf_model.feature_importances_
            
            # Get top 5
            top_indices = np.argsort(feature_importance)[-5:][::-1]
            
            for idx in top_indices:
                feature_name = feature_names[idx]
                importance = feature_importance[idx]
                feature_value = input_data.iloc[0, idx]
                
                st.markdown(f"""
                <div class="feature-contribution">
                    <strong>{feature_name}:</strong> {feature_value} 
                    <span class="feature-meta">(Importance: {importance*100:.1f}%)</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Feature importance chart
        st.markdown("---")
        st.subheader("📈 Feature Importance Analysis")
        render_feature_importance_visual(
            feature_names,
            input_prepared.iloc[0].to_numpy(),
            rf_model.feature_importances_
        )
        
        # Clinical recommendations
        st.markdown("---")
        st.subheader("💡 Clinical Recommendations")
        
        recommendation = get_clinical_recommendation(risk_level, ensemble_prob)
        st.markdown(recommendation)
        
        # Additional information
        with st.expander("ℹ️ Understanding Your Results"):
            st.markdown("""
            ### How to interpret these results:
            
            **Risk Classification:**
            - The system classifies pregnancies as either "High Risk" or "Low Risk"
            - Classification is based on the ensemble probability (average of both models)
            - Threshold: 50% probability for high-risk classification
            
            **Probability Scores:**
            - Show the confidence level of each model's prediction
            - Higher percentages indicate stronger confidence
            - Ensemble combines both models for more robust assessment
            
            **Model Agreement:**
            - Strong agreement (< 10% difference) indicates high confidence
            - Lower agreement may warrant additional clinical review
            
            **Contributing Factors:**
            - Shows which features most influenced the prediction
            - Helps identify specific risk factors for intervention
            - Based on Random Forest feature importance
            
            **Important Notes:**
            - This system provides **decision support**, not diagnosis
            - Healthcare provider judgment is essential
            - Use results to inform, not replace, clinical assessment
            - Consider additional factors not captured by the model
            """)
        
        # Export option
        st.markdown("---")
        report = f"""
PREGNANCY RISK ASSESSMENT REPORT
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

PATIENT INFORMATION:
- Age: {age} years
- Systolic BP: {systolic_bp} mmHg
- Diastolic BP: {diastolic_bp} mmHg
- Blood Sugar: {blood_sugar} mmol/L
- Body Temperature: {body_temp} °F
- BMI: {bmi}
- Heart Rate: {heart_rate} bpm
- Previous Complications: {'Yes' if previous_complications else 'No'}
- Preexisting Diabetes: {'Yes' if preexisting_diabetes else 'No'}
- Gestational Diabetes: {'Yes' if gestational_diabetes else 'No'}
- Mental Health Concerns: {'Yes' if mental_health else 'No'}

RISK ASSESSMENT:
- Final Classification: {risk_level}
- Ensemble Probability: {ensemble_prob*100:.2f}%
- Random Forest Probability: {rf_prob*100:.2f}%
- XGBoost Probability: {xgb_prob*100:.2f}%
- Model Agreement: {agreement_status}

TOP CONTRIBUTING FACTORS:
"""
        for idx in top_indices:
            report += f"- {feature_names[idx]}: {input_data.iloc[0, idx]}\n"
        
        report += f"\nRECOMMENDATION:\n{recommendation}"
        
        st.download_button(
            label="📄 Download Detailed Report",
            data=report,
            file_name=f"pregnancy_risk_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            on_click="ignore"
        )


if __name__ == "__main__":
    main()