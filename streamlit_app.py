import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Set page config
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Advanced Dark Theme UI
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 100%);
        color: #ffffff;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
        color: #ffffff;
    }

    .stButton>button {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%) !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: bold !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'heart_disease_nb_model.pkl')
        scaler_path = os.path.join(script_dir, 'scaler.pkl')
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

model, scaler = load_model_and_scaler()

# Define mappings
cp_mapping = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}

restecg_mapping = {
    "Normal": 0,
    "ST-T Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}

slope_mapping = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}

thal_mapping = {
    "Normal": 0,
    "Fixed Defect": 1,
    "Reversible Defect": 2
}

# Sidebar
with st.sidebar:
    st.markdown("### 🏥 Heart Disease Predictor")
    st.markdown("---")
    page = st.radio("Select Page", ["🏠 Home", "🔮 Prediction", "📊 About"])

if page == "🏠 Home":
    st.markdown("""
    <div style='text-align: center; padding: 40px 20px;'>
    <h1 style='font-size: 48px; margin: 20px 0;'>❤️ Heart Disease Prediction System</h1>
    <p style='font-size: 20px; color: #e74c3c;'>Advanced AI-Powered Medical Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("Use the Prediction page to analyze heart disease risk based on medical parameters.")

elif page == "🔮 Prediction":
    st.markdown("""
    <div style='text-align: center; padding: 30px 20px;'>
    <h1 style='color: #e74c3c;'>🔮 Heart Disease Risk Assessment</h1>
    </div>
    """, unsafe_allow_html=True)
    
    if model is None or scaler is None:
        st.error("❌ Model not loaded. Please refresh the page.")
    else:
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input("👶 Age", min_value=1, max_value=120, value=50)
                sex = st.radio("👥 Sex", ["Male", "Female"], horizontal=True)
                cp = st.selectbox("💔 Chest Pain Type", list(cp_mapping.keys()))
            
            with col2:
                trestbps = st.number_input("🩸 Blood Pressure", min_value=50, max_value=200, value=120)
                chol = st.number_input("🧬 Cholesterol", min_value=100, max_value=600, value=200)
                thalach = st.number_input("❤️ Max Heart Rate", min_value=60, max_value=220, value=150)
            
            with col3:
                fbs = st.radio("🍬 Fasting Blood Sugar > 120", ["No", "Yes"], horizontal=True)
                restecg = st.selectbox("📈 Resting ECG", list(restecg_mapping.keys()))
                exang = st.radio("🏃 Exercise Angina", ["No", "Yes"], horizontal=True)
            
            col4, col5, col6, col7 = st.columns(4)
            
            with col4:
                oldpeak = st.number_input("📉 ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            
            with col5:
                slope = st.selectbox("⬆️ ST Segment Slope", list(slope_mapping.keys()))
            
            with col6:
                ca = st.number_input("🚢 Major Vessels", min_value=0, max_value=4, value=0)
            
            with col7:
                thal = st.selectbox("🩸 Thalassemia", list(thal_mapping.keys()))
            
            submitted = st.form_submit_button("🔮 Predict", use_container_width=True)
        
        if submitted:
            try:
                input_data = pd.DataFrame({
                    'age': [age],
                    'sex': [1 if sex == "Male" else 0],
                    'cp': [cp_mapping[cp]],
                    'trestbps': [trestbps],
                    'chol': [chol],
                    'fbs': [1 if fbs == "Yes" else 0],
                    'restecg': [restecg_mapping[restecg]],
                    'thalach': [thalach],
                    'exang': [1 if exang == "Yes" else 0],
                    'oldpeak': [oldpeak],
                    'slope': [slope_mapping[slope]],
                    'ca': [ca],
                    'thal': [thal_mapping[thal]]
                })
                
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0]
                
                st.markdown("---")
                st.markdown("### 📋 Results")
                
                risk_percentage = probability[1] * 100
                
                if prediction == 1:
                    st.error(f"⚠️ HIGH RISK: {risk_percentage:.1f}%")
                    st.warning("Please consult a healthcare professional immediately.")
                else:
                    st.success(f"✅ LOW RISK: {risk_percentage:.1f}%")
                    st.info("Continue healthy lifestyle habits.")
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

elif page == "📊 About":
    st.markdown("""
    ### About Heart Disease
    
    Heart disease is the leading cause of death worldwide. This AI model uses medical parameters to assess risk.
    
    **Disclaimer:** This tool is for educational purposes only and should NOT replace professional medical diagnosis.
    """)
