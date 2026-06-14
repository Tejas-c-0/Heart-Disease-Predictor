# ❤️ Heart Disease Prediction System

An advanced AI-powered web application for predicting heart disease risk using Machine Learning and Streamlit.

## 🌐 Live Demo
**[Deploy on Streamlit Cloud](https://share.streamlit.io)** - Click to access the live application

## ✨ Features

- **🏠 Home Page**: Introduction and application overview
- **🔮 Prediction Page**: Interactive form for heart disease risk assessment
- **📊 About Page**: Information about heart disease and risk factors
- **🤖 AI Model**: Gaussian Naive Bayes classifier trained on UCI Heart Disease Dataset
- **📈 Accurate Predictions**: 85%+ accuracy with scaled input data
- **🎨 Modern UI**: Dark theme with gradient design and responsive layout

## 🛠️ Tech Stack

- **Framework**: Streamlit
- **ML Algorithm**: Scikit-learn (Gaussian Naive Bayes)
- **Data Processing**: Pandas, NumPy
- **Model Persistence**: Joblib
- **Deployment**: Streamlit Cloud

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- Git

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/Project-3-Heart.git
cd Project-3-Heart
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

## 🚀 Deployment to Streamlit Cloud

1. **Push code to GitHub**
```bash
git add .
git commit -m "Deploy: Heart Disease Predictor"
git push origin main
```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository and `streamlit_app.py`
   - Click "Deploy"

## 📖 How to Use

### 1. **Home Page**
   - View application overview
   - Understand the technology and purpose
   - Read important medical disclaimer

### 2. **Prediction Page**
   Enter your medical information:
   
   **Basic Information:**
   - Age (years)
   - Sex (Male/Female)
   - Chest Pain Type
   
   **Cardiovascular Measurements:**
   - Resting Blood Pressure (mm Hg)
   - Serum Cholesterol (mg/dl)
   - Maximum Heart Rate Achieved
   
   **Blood & ECG Analysis:**
   - Fasting Blood Sugar
   - Resting ECG Results
   - Exercise Induced Angina
   
   **Advanced Parameters:**
   - ST Depression
   - ST Segment Slope
   - Major Vessels
   - Thalassemia Type

3. **View Results**
   - Risk percentage
   - Prediction (High Risk / Low Risk)
   - Health recommendations

## 📊 Model Information

- **Dataset**: UCI Heart Disease Dataset
- **Samples**: 303 patients
- **Features**: 13 medical parameters
- **Algorithm**: Gaussian Naive Bayes
- **Data Preprocessing**: StandardScaler normalization
- **Accuracy**: Tested on validation set

## 📁 Project Structure

```
Project-3-Heart/
├── streamlit_app.py              # Main application
├── requirements.txt              # Dependencies
├── runtime.txt                   # Python version
├── heart.csv                     # Dataset
├── heart_disease_nb_model.pkl    # Trained model
├── scaler.pkl                    # Data scaler
├── README.md                     # Documentation
├── .gitignore                    # Git ignore rules
└── .streamlit/
    └── config.toml               # Streamlit config
```

## ⚠️ Medical Disclaimer

**IMPORTANT**: This application is for **educational purposes only** and should **NOT** be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical advice.

## 🔐 Privacy & Security

- No data is stored on servers
- All computations happen locally
- No personal information is collected
- Model predictions are real-time only

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Submit pull requests

## 📝 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Created as a machine learning educational project.

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Disclaimer**: This tool is for educational purposes. Always seek professional medical advice.

### 2. **Prediction Page**
   - Enter patient medical information
   - Fill in all required fields:
     - Age (1-120 years)
     - Sex (Male/Female)
     - Chest Pain Type
     - Resting Blood Pressure
     - Serum Cholesterol
     - Fasting Blood Sugar
     - Resting ECG Results
     - Maximum Heart Rate
     - Exercise Induced Angina
     - ST Depression
     - ST Segment Slope
     - Major Vessels Count
     - Thalassemia Type
   - Click "Predict" button
   - View risk assessment and recommendations

### 3. **About Page**
   - Learn about heart disease
   - Understand common symptoms
   - Review risk factors
   - Get prevention tips

### 4. **Dataset Page**
   - View dataset statistics
   - Check feature descriptions
   - See sample data
   - Review statistical summaries

## 📊 Dataset Information

The model is trained on the UCI Heart Disease dataset with:
- **Records**: 303 patients
- **Features**: 13 medical parameters
- **Classes**: 2 (Heart Disease / No Disease)

### Feature Descriptions

| Feature | Description |
|---------|-------------|
| age | Age in years |
| sex | Sex (1 = Male, 0 = Female) |
| cp | Chest pain type (0-3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 (1 = Yes, 0 = No) |
| restecg | Resting ECG results (0-2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina (1 = Yes, 0 = No) |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment (0-2) |
| ca | Number of major vessels (0-4) |
| thal | Thalassemia (0-2) |

## ⚠️ Important Disclaimer

**This application is for educational and informational purposes only.**

- The predictions made by this application should NOT be used as a substitute for professional medical advice, diagnosis, or treatment.
- Always consult with a qualified healthcare professional for any medical concerns.
- This tool is not FDA-approved and does not replace clinical judgment.
- Use this application at your own risk.

## 🔍 Model Details

- **Algorithm**: Naive Bayes Classifier
- **Trained on**: UCI Heart Disease Dataset
- **Training samples**: 303 patients
- **Features used**: 13 medical parameters

## 🛠️ Troubleshooting

### Error: Model file not found
**Solution**: Ensure `heart_disease_nb_model.pkl` is in the same directory as `app.py`

### Error: Dataset not found
**Solution**: Ensure `heart.csv` is in the same directory as `app.py`

### Port already in use
**Solution**: Use a different port:
```bash
streamlit run app.py --server.port 8502
```

### Module import errors
**Solution**: Reinstall dependencies:
```bash
pip install --upgrade -r requirements.txt
```

## 📁 Project Structure

```
Project-3 Heart/
├── app.py                           # Main Streamlit application
├── Heart Disease Predict.py         # Original model script
├── Heart Disease.ipynb              # Jupyter notebook
├── heart.csv                        # Dataset
├── heart_disease_nb_model.pkl       # Trained model
└── requirements.txt                 # Dependencies
```

## 💡 Tips

1. **For best results**: Ensure all medical parameters are accurate
2. **High risk results**: If the model predicts high risk, please consult a healthcare professional immediately
3. **Regular check-ups**: Get periodic health evaluations regardless of prediction results
4. **Data privacy**: This application doesn't store or transmit user data

## 🤝 Contributing

Feel free to improve this application by:
- Adding more features
- Improving the UI/UX
- Adding more advanced models
- Enhancing visualizations

## 📝 License

This project is for educational purposes.

## 👨‍💻 Author

Created as a Heart Disease Prediction Project

---

**Remember**: Your health is your wealth. Always prioritize professional medical advice! ❤️
