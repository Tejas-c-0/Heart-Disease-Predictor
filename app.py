import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Clear cache to refresh model on startup
st.cache_resource.clear()

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
    
    .main {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 100%);
        padding: 20px;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 100%);
    }
    
    .stTabs [role="tablist"] button {
        color: #ffffff !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        margin: 5px !important;
    }
    
    .stTabs [role="tablist"] button[aria-selected="true"] {
        background-color: #e74c3c !important;
        color: #ffffff !important;
    }
    
    .stContainer {
        background: linear-gradient(135deg, rgba(30, 30, 50, 0.9) 0%, rgba(20, 20, 40, 0.9) 100%);
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        border: 1px solid rgba(231, 76, 60, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%) !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: bold !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(231, 76, 60, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #c0392b 0%, #a93226 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(231, 76, 60, 0.6) !important;
    }
    
    .stTextInput>div>div>input {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
        border: 1px solid rgba(231, 76, 60, 0.3) !important;
        border-radius: 8px !important;
    }
    
    .stNumberInput>div>div>input {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
        border: 1px solid rgba(231, 76, 60, 0.3) !important;
        border-radius: 8px !important;
    }
    
    .stSelectbox>div>div>select {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
        border: 1px solid rgba(231, 76, 60, 0.3) !important;
        border-radius: 8px !important;
    }
    
    .stRadio>div>div>label {
        color: #ffffff !important;
    }
    
    .prediction-high {
        background: linear-gradient(135deg, rgba(244, 67, 54, 0.2) 0%, rgba(211, 47, 47, 0.2) 100%);
        border-left: 5px solid #f44336;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        color: #ffffff;
        box-shadow: 0 4px 15px rgba(244, 67, 54, 0.3);
    }
    
    .prediction-low {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.2) 0%, rgba(56, 142, 60, 0.2) 100%);
        border-left: 5px solid #4caf50;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        color: #ffffff;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(33, 150, 243, 0.15) 0%, rgba(13, 71, 161, 0.15) 100%);
        border-left: 5px solid #2196f3;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        color: #ffffff;
        box-shadow: 0 4px 15px rgba(33, 150, 243, 0.2);
    }
    
    .heart-beat {
        animation: heartbeat 1.5s ease-in-out infinite;
    }
    
    @keyframes heartbeat {
        0%, 100% { transform: scale(1); }
        14% { transform: scale(1.1); }
        28% { transform: scale(1); }
        42% { transform: scale(1.1); }
        70% { transform: scale(1); }
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(231, 76, 60, 0.2) 0%, rgba(192, 57, 43, 0.2) 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px;
        border: 1px solid rgba(231, 76, 60, 0.3);
        color: #ffffff;
    }
    
    .footer-text {
        text-align: center;
        color: #888888;
        font-size: 12px;
        margin-top: 30px;
        padding: 20px;
        border-top: 1px solid rgba(231, 76, 60, 0.2);
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    .stMetric {
        background: linear-gradient(135deg, rgba(231, 76, 60, 0.1) 0%, rgba(192, 57, 43, 0.1) 100%);
        border-radius: 10px;
        padding: 15px;
        border: 1px solid rgba(231, 76, 60, 0.2);
    }
    
    .stMetric label {
        color: #b0b0b0 !important;
    }
    
    .stMetric [data-testid="metricValue"] {
        color: #e74c3c !important;
    }
</style>
""", unsafe_allow_html=True)

# Load model and scaler with error handling
def load_model_and_scaler():
    if 'model' not in st.session_state or 'scaler' not in st.session_state:
        try:
            # Get the directory of the current script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, 'heart_disease_nb_model.pkl')
            scaler_path = os.path.join(script_dir, 'scaler.pkl')
            
            if not os.path.exists(model_path):
                st.error(f"❌ Model file not found at: {model_path}")
                st.session_state.model = None
                st.session_state.scaler = None
                return None, None
            
            if not os.path.exists(scaler_path):
                st.error(f"❌ Scaler file not found at: {scaler_path}")
                st.session_state.model = None
                st.session_state.scaler = None
                return None, None
            
            st.session_state.model = joblib.load(model_path)
            st.session_state.scaler = joblib.load(scaler_path)
        except Exception as e:
            st.error(f"❌ Error loading model or scaler: {str(e)}")
            st.session_state.model = None
            st.session_state.scaler = None
            return None, None
    
    return st.session_state.model, st.session_state.scaler

model, scaler = load_model_and_scaler()

# Define mappings for categorical variables
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
    page = st.radio("Select Page", ["🏠 Home", "🔮 Prediction", "📚 About Heart Disease", "📊 Dataset Info"])

# Home Page
if page == "🏠 Home":
    st.markdown("""
    <div style='text-align: center; padding: 40px 20px;'>
    <h1 style='font-size: 48px; margin: 20px 0;'>❤️ Heart Disease Prediction System</h1>
    <p style='font-size: 20px; color: #e74c3c; margin: 10px 0;'>Advanced AI-Powered Medical Analysis</p>
    <p style='font-size: 14px; color: #888888;'>Early Detection Using Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display heart images
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='text-align: center;'>
        <img src='https://cdn-icons-png.flaticon.com/512/2913/2913152.png' width='120' style='margin: 10px 0; filter: drop-shadow(0 0 10px rgba(231, 76, 60, 0.5));'>
        <h3 style='color: #e74c3c;'>AI Powered</h3>
        <p style='color: #888888;'>Advanced Machine Learning</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center;'>
        <img src='https://cdn-icons-png.flaticon.com/512/2913/2913254.png' width='120' style='margin: 10px 0; filter: drop-shadow(0 0 10px rgba(231, 76, 60, 0.5));'>
        <h3 style='color: #e74c3c;'>Real-time Analysis</h3>
        <p style='color: #888888;'>Instant Results</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center;'>
        <img src='https://cdn-icons-png.flaticon.com/512/2913/2913190.png' width='120' style='margin: 10px 0; filter: drop-shadow(0 0 10px rgba(231, 76, 60, 0.5));'>
        <h3 style='color: #e74c3c;'>Secure & Safe</h3>
        <p style='color: #888888;'>Privacy First</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='info-box'>
        <h3 style='color: #e74c3c;'>🎯 Purpose</h3>
        <p>This application uses advanced Machine Learning to predict heart disease risk based on medical parameters with high accuracy.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-box'>
        <h3 style='color: #e74c3c;'>🔬 Technology Stack</h3>
        <ul>
        <li>Streamlit - Interactive Web Interface</li>
        <li>Scikit-learn - ML Framework</li>
        <li>Naive Bayes Algorithm - Classification</li>
        <li>Pandas & NumPy - Data Processing</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-box'>
        <h3 style='color: #e74c3c;'>📊 Model Accuracy</h3>
        <p>Trained on UCI Heart Disease Dataset with 303 patient records. The model uses 13 advanced medical features for accurate predictions.</p>
        <p style='margin-top: 10px;'><strong>Features:</strong> Age, Sex, Blood Pressure, Cholesterol, ECG Results, and more...</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-box'>
        <h3 style='color: #e74c3c;'>⚠️ Medical Disclaimer</h3>
        <p>This tool is for educational purposes only. Not a substitute for professional medical diagnosis. Always consult healthcare professionals.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; margin: 30px 0;'>
    <h2 style='color: #e74c3c;'>Quick Start Guide</h2>
    <div style='display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 15px; margin: 20px 0;'>
        <div style='background: rgba(231, 76, 60, 0.1); padding: 15px; border-radius: 10px;'>
            <h4 style='color: #e74c3c;'>Step 1</h4>
            <p style='color: #888888;'>Go to Prediction</p>
        </div>
        <div style='background: rgba(231, 76, 60, 0.1); padding: 15px; border-radius: 10px;'>
            <h4 style='color: #e74c3c;'>Step 2</h4>
            <p style='color: #888888;'>Enter Medical Data</p>
        </div>
        <div style='background: rgba(231, 76, 60, 0.1); padding: 15px; border-radius: 10px;'>
            <h4 style='color: #e74c3c;'>Step 3</h4>
            <p style='color: #888888;'>Click Predict</p>
        </div>
        <div style='background: rgba(231, 76, 60, 0.1); padding: 15px; border-radius: 10px;'>
            <h4 style='color: #e74c3c;'>Step 4</h4>
            <p style='color: #888888;'>Get Results</p>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

# Prediction Page
elif page == "🔮 Prediction":
    st.markdown("""
    <div style='text-align: center; padding: 30px 20px;'>
    <h1 style='color: #e74c3c;'>🔮 Heart Disease Risk Assessment</h1>
    <p style='color: #888888;'>Advanced Medical Analysis Using AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    <h3 style='color: #e74c3c;'>📋 Instructions</h3>
    <p>Please enter accurate medical information for the most reliable prediction. All fields are required.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if model is None or scaler is None:
        st.error("❌ Cannot proceed without the model or scaler. Please check if 'heart_disease_nb_model.pkl' and 'scaler.pkl' exist.")
    else:
        with st.form("prediction_form", clear_on_submit=False):
            # Section 1: Basic Patient Information
            st.markdown("""
            <div style='background: linear-gradient(135deg, rgba(255, 107, 107, 0.15), rgba(255, 160, 122, 0.15)); 
                        border-radius: 15px; padding: 25px; margin: 15px 0; border-left: 5px solid #ff6b6b;'>
            <h2 style='color: #ff6b6b; margin: 0 0 20px 0;'>👤 Basic Patient Information</h2>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("<div style='padding: 20px; background: linear-gradient(135deg, rgba(255, 107, 107, 0.1), rgba(255, 160, 122, 0.1)); border-radius: 10px; border: 2px solid rgba(255, 107, 107, 0.3);'>", unsafe_allow_html=True)
                age = st.number_input("👶 Age (years)", min_value=1, max_value=120, value=50, step=1, help="Patient age in years")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div style='padding: 20px; background: linear-gradient(135deg, rgba(255, 160, 122, 0.1), rgba(255, 200, 124, 0.1)); border-radius: 10px; border: 2px solid rgba(255, 160, 122, 0.3);'>", unsafe_allow_html=True)
                sex = st.radio("👥 Sex", ["Male", "Female"], horizontal=True, help="Patient sex/gender")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col3:
                st.markdown("<div style='padding: 20px; background: linear-gradient(135deg, rgba(255, 200, 124, 0.1), rgba(255, 232, 31, 0.1)); border-radius: 10px; border: 2px solid rgba(255, 200, 124, 0.3);'>", unsafe_allow_html=True)
                cp = st.selectbox("💔 Chest Pain Type", list(cp_mapping.keys()), help="Type of chest pain experienced")
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Section 2: Cardiovascular Measurements
            st.markdown("""
            <div style='background: linear-gradient(135deg, rgba(76, 175, 80, 0.15), rgba(139, 195, 74, 0.15)); 
                        border-radius: 15px; padding: 25px; margin: 15px 0; border-left: 5px solid #4caf50;'>
            <h2 style='color: #4caf50; margin: 0 0 20px 0;'>💚 Cardiovascular Measurements</h2>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("<div style='padding: 20px; background: linear-gradient(135deg, rgba(76, 175, 80, 0.1), rgba(102, 187, 106, 0.1)); border-radius: 10px; border: 2px solid rgba(76, 175, 80, 0.3);'>", unsafe_allow_html=True)
                trestbps = st.number_input("🩸 Resting Blood Pressure", min_value=50, max_value=200, value=120, step=1, help="Resting blood pressure in mm Hg")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div style='padding: 20px; background: linear-gradient(135deg, rgba(102, 187, 106, 0.1), rgba(129, 199, 132, 0.1)); border-radius: 10px; border: 2px solid rgba(102, 187, 106, 0.3);'>", unsafe_allow_html=True)
                chol = st.number_input("🧬 Serum Cholesterol", min_value=100, max_value=600, value=200, step=1, help="Serum cholesterol level in mg/dl")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col3:
                st.markdown("<div style='padding: 20px; background: linear-gradient(135deg, rgba(129, 199, 132, 0.1), rgba(156, 204, 101, 0.1)); border-radius: 10px; border: 2px solid rgba(129, 199, 132, 0.3);'>", unsafe_allow_html=True)
                thalach = st.number_input("❤️ Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150, step=1, help="Maximum heart rate achieved during exercise")
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Section 3: Blood & ECG Analysis
            st.markdown("""
            <div style='background: linear-gradient(135deg, rgba(33, 150, 243, 0.15), rgba(66, 165, 245, 0.15)); 
                        border-radius: 15px; padding: 25px; margin: 15px 0; border-left: 5px solid #2196f3;'>
            <h2 style='color: #2196f3; margin: 0 0 20px 0;'>🔬 Blood & ECG Analysis</h2>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("<div style='padding: 20px; background: linear-gradient(135deg, rgba(33, 150, 243, 0.1), rgba(66, 165, 245, 0.1)); border-radius: 10px; border: 2px solid rgba(33, 150, 243, 0.3);'>", unsafe_allow_html=True)
                fbs = st.radio("🍬 Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"], horizontal=True, help="Is fasting blood sugar greater than 120?")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div style='padding: 20px; background: linear-gradient(135deg, rgba(66, 165, 245, 0.1), rgba(100, 181, 246, 0.1)); border-radius: 10px; border: 2px solid rgba(66, 165, 245, 0.3);'>", unsafe_allow_html=True)
                restecg = st.selectbox("📈 Resting ECG Results", list(restecg_mapping.keys()), help="Resting electrocardiography results")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col3:
                st.markdown("<div style='padding: 20px; background: linear-gradient(135deg, rgba(100, 181, 246, 0.1), rgba(144, 202, 249, 0.1)); border-radius: 10px; border: 2px solid rgba(100, 181, 246, 0.3);'>", unsafe_allow_html=True)
                exang = st.radio("🏃 Exercise Induced Angina", ["No", "Yes"], horizontal=True, help="Angina induced by exercise?")
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Section 4: Advanced Parameters
            st.markdown("""
            <div style='background: linear-gradient(135deg, rgba(156, 39, 176, 0.15), rgba(186, 104, 200, 0.15)); 
                        border-radius: 15px; padding: 25px; margin: 15px 0; border-left: 5px solid #9c27b0;'>
            <h2 style='color: #9c27b0; margin: 0 0 20px 0;'>⚙️ Advanced Exercise Parameters</h2>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("<div style='padding: 20px; background: linear-gradient(135deg, rgba(156, 39, 176, 0.1), rgba(171, 71, 188, 0.1)); border-radius: 10px; border: 2px solid rgba(156, 39, 176, 0.3);'>", unsafe_allow_html=True)
                oldpeak = st.number_input("📉 ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1, help="ST depression induced by exercise")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div style='padding: 20px; background: linear-gradient(135deg, rgba(171, 71, 188, 0.1), rgba(186, 104, 200, 0.1)); border-radius: 10px; border: 2px solid rgba(171, 71, 188, 0.3);'>", unsafe_allow_html=True)
                slope = st.selectbox("⬆️ ST Segment Slope", list(slope_mapping.keys()), help="Slope of peak exercise ST segment")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col3:
                st.markdown("<div style='padding: 20px; background: linear-gradient(135deg, rgba(186, 104, 200, 0.1), rgba(206, 147, 216, 0.1)); border-radius: 10px; border: 2px solid rgba(186, 104, 200, 0.3);'>", unsafe_allow_html=True)
                ca = st.number_input("🚢 Major Vessels", min_value=0, max_value=4, value=0, step=1, help="Major vessels colored by fluoroscopy")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col4:
                st.markdown("<div style='padding: 20px; background: linear-gradient(135deg, rgba(206, 147, 216, 0.1), rgba(225, 190, 231, 0.1)); border-radius: 10px; border: 2px solid rgba(206, 147, 216, 0.3);'>", unsafe_allow_html=True)
                thal = st.selectbox("🩸 Thalassemia Type", list(thal_mapping.keys()), help="Type of thalassemia detected")
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            submitted = st.form_submit_button("🔮 Analyze & Predict", use_container_width=True)
        
        if submitted:
            # Debug: Check model and scaler status
            if model is None or scaler is None:
                st.error("❌ Error: Model or scaler is not loaded. Please refresh the page.")
                st.stop()
            
            # Prepare input data
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
            
            # Make prediction
            try:
                with st.spinner('🔄 Analyzing medical data...'):
                    if not hasattr(model, 'predict'):
                        st.error(f"❌ Error: Model object is invalid. Type: {type(model)}")
                        st.stop()
                    
                    # Scale the input data using the saved scaler
                    input_data_scaled = scaler.transform(input_data)
                    
                    prediction = model.predict(input_data_scaled)[0]
                    probability = model.predict_proba(input_data_scaled)[0]
                    import time
                    time.sleep(1)  # For visual effect
                
                st.markdown("---")
                st.markdown("### 📋 Prediction Results")
                
                if prediction == 1:
                    risk_percentage = probability[1] * 100
                    st.markdown(f"""
                    <div class='prediction-high'>
                    <h2 style='color: #f44336;'>⚠️ High Risk Detected</h2>
                    <p style='font-size: 32px; font-weight: bold; color: #e74c3c;'>{risk_percentage:.1f}%</p>
                    <p><strong>Risk Assessment:</strong> Potential heart disease indicators detected</p>
                    <p style='margin-top: 10px; color: #aaa;'>Status: Requires immediate medical attention</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class='info-box' style='border-left: 5px solid #f44336;'>
                    <h3 style='color: #f44336;'>🏥 Urgent Recommendations:</h3>
                    <ul style='color: #ffffff;'>
                    <li><b>Schedule a cardiology appointment immediately</b></li>
                    <li>Undergo comprehensive cardiac testing (ECG, Echocardiogram, Stress Test)</li>
                    <li>Adopt a strict heart-healthy diet (Mediterranean diet recommended)</li>
                    <li>Start a supervised exercise program</li>
                    <li>Take prescribed medications regularly</li>
                    <li>Monitor blood pressure and cholesterol weekly</li>
                    <li>Reduce stress through meditation/yoga</li>
                    <li>Avoid smoking and alcohol</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    risk_percentage = probability[1] * 100
                    st.markdown(f"""
                    <div class='prediction-low'>
                    <h2 style='color: #4caf50;'>✅ Low Risk Detected</h2>
                    <p style='font-size: 32px; font-weight: bold; color: #66bb6a;'>{risk_percentage:.1f}%</p>
                    <p><strong>Risk Assessment:</strong> Currently low risk of heart disease</p>
                    <p style='margin-top: 10px; color: #aaa;'>Status: Continue healthy lifestyle</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class='info-box' style='border-left: 5px solid #4caf50;'>
                    <h3 style='color: #4caf50;'>💪 Health Maintenance Tips:</h3>
                    <ul style='color: #ffffff;'>
                    <li>Continue regular exercise (30 minutes daily)</li>
                    <li>Maintain a balanced, nutritious diet</li>
                    <li>Keep stress levels low through relaxation techniques</li>
                    <li>Get adequate sleep (7-9 hours per night)</li>
                    <li>Monitor blood pressure and cholesterol annually</li>
                    <li>Avoid smoking and excessive alcohol consumption</li>
                    <li>Schedule regular health check-ups (annually)</li>
                    <li>Stay hydrated and limit sodium intake</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.warning("⚠️ **Medical Disclaimer:** This AI prediction is for educational purposes only and should NOT replace professional medical diagnosis. Always consult qualified healthcare professionals.")
            
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")

# About Heart Disease Page
elif page == "📚 About Heart Disease":
    st.markdown("""
    <div style='text-align: center; padding: 30px 20px;'>
    <h1 style='color: #e74c3c;'>📚 Understanding Heart Disease</h1>
    <p style='color: #888888;'>Comprehensive Medical Information & Prevention Guide</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='info-box'>
        <h3>What is Heart Disease?</h3>
        <p>Heart disease is a broad term that includes several types of conditions affecting the heart. It's the leading cause of death worldwide. Common types include:</p>
        <ul>
        <li><b>Coronary Artery Disease:</b> Narrowing of arteries supplying the heart</li>
        <li><b>Heart Failure:</b> Heart's inability to pump blood effectively</li>
        <li><b>Arrhythmias:</b> Irregular heartbeats</li>
        <li><b>Valvular Disease:</b> Problems with heart valves</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-box'>
        <h3>⚠️ Common Symptoms</h3>
        <p>Be alert to these warning signs:</p>
        <ul>
        <li>Chest pain or discomfort</li>
        <li>Shortness of breath</li>
        <li>Fatigue or weakness</li>
        <li>Heart palpitations</li>
        <li>Dizziness or fainting</li>
        <li>Nausea or cold sweats</li>
        <li>Pain in arm, neck, or jaw</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("""
        <div class='info-box'>
        <h3>🚨 Risk Factors</h3>
        <ul>
        <li><b>Age:</b> Risk increases with age</li>
        <li><b>Gender:</b> Men at higher risk</li>
        <li><b>Family History:</b> Genetic predisposition</li>
        <li><b>High Blood Pressure:</b> Damages arteries</li>
        <li><b>High Cholesterol:</b> Builds up in arteries</li>
        <li><b>Smoking:</b> Increases risk significantly</li>
        <li><b>Diabetes:</b> Increases heart disease risk</li>
        <li><b>Obesity:</b> Extra weight strains heart</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='info-box'>
        <h3>💚 Prevention Tips</h3>
        <ul>
        <li><b>Exercise:</b> 150 mins moderate activity weekly</li>
        <li><b>Diet:</b> Mediterranean or DASH diet</li>
        <li><b>Weight:</b> Maintain healthy BMI</li>
        <li><b>Stress:</b> Practice relaxation techniques</li>
        <li><b>Sleep:</b> Get 7-9 hours daily</li>
        <li><b>Quit Smoking:</b> Critical for heart health</li>
        <li><b>Limit Alcohol:</b> Moderate consumption only</li>
        <li><b>Regular Check-ups:</b> Monitor vital signs</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# Dataset Info Page
elif page == "📊 Dataset Info":
    st.markdown("""
    <div style='text-align: center; padding: 30px 20px;'>
    <h1 style='color: #e74c3c;'>📊 Dataset Information & Statistics</h1>
    <p style='color: #888888;'>UCI Heart Disease Dataset Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    <h3 style='color: #e74c3c;'>📋 Dataset Overview</h3>
    <p>This model is trained on the UCI Heart Disease dataset containing 303 patient records with 13 medical features.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load and display dataset
    try:
        df = pd.read_csv('heart.csv')
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Features", len(df.columns) - 1)
        with col3:
            st.metric("Disease Cases", (df['target'] == 1).sum())
        with col4:
            st.metric("Healthy Cases", (df['target'] == 0).sum())
        
        st.markdown("### 📋 Feature Descriptions")
        
        features_desc = {
            "age": "Age in years",
            "sex": "Sex (1 = male, 0 = female)",
            "cp": "Chest pain type (0-3)",
            "trestbps": "Resting blood pressure (mm Hg)",
            "chol": "Serum cholesterol (mg/dl)",
            "fbs": "Fasting blood sugar > 120 (1 = yes, 0 = no)",
            "restecg": "Resting ECG results (0-2)",
            "thalach": "Maximum heart rate achieved",
            "exang": "Exercise induced angina (1 = yes, 0 = no)",
            "oldpeak": "ST depression induced by exercise",
            "slope": "Slope of peak exercise ST segment (0-2)",
            "ca": "Number of major vessels (0-4)",
            "thal": "Thalassemia (0-2)",
            "target": "Heart disease (1 = yes, 0 = no)"
        }
        
        for feature, description in features_desc.items():
            st.write(f"**{feature}:** {description}")
        
        st.markdown("### 📈 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("### 📊 Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        
    except FileNotFoundError:
        st.error("❌ Dataset file 'heart.csv' not found!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; margin: 30px 0; padding: 30px; border-top: 2px solid rgba(231, 76, 60, 0.2);'>
<h3 style='color: #e74c3c; margin: 10px 0;'>❤️ Heart Disease Prediction System</h3>
<p style='color: #888888; font-size: 14px; margin: 5px 0;'>Advanced AI-Powered Medical Analysis Platform</p>
<p style='color: #666666; font-size: 13px; margin: 15px 0;'>Powered by Machine Learning | Built with Streamlit | Data Science Innovation</p>
<p style='color: #e74c3c; font-size: 16px; font-weight: bold; margin: 15px 0;'>🎓 Created by <span style='color: #ffffff; background: linear-gradient(135deg, #e74c3c, #c0392b); padding: 2px 8px; border-radius: 4px;'>Tejas</span></p>
<p style='color: #888888; font-size: 11px; margin-top: 15px;'>⚠️ For educational and informational purposes only. Not a substitute for professional medical advice.</p>
<p style='color: #666666; font-size: 11px; margin-top: 10px;'>© 2026 - All Rights Reserved | Disclaimer: This application provides predictions based on machine learning models and should not be used for actual medical diagnosis.</p>
</div>
""", unsafe_allow_html=True)
