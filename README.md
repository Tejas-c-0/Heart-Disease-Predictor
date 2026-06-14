# ❤️ Heart Disease Prediction System

An advanced web-based application for predicting heart disease risk using Machine Learning.

## 📋 Features

- **🏠 Home Page**: Introduction and application overview
- **🔮 Prediction Page**: Interactive form for heart disease risk prediction
- **📚 About Page**: Detailed information about heart disease, symptoms, and risk factors
- **📊 Dataset Page**: View dataset statistics and feature descriptions
- **🔒 Secure Model**: Uses trained Naive Bayes model for accurate predictions

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Verify Files
Make sure you have the following files in the project directory:
- `app.py` - Main Streamlit application
- `heart_disease_nb_model.pkl` - Trained model file
- `heart.csv` - Dataset
- `requirements.txt` - Dependencies file

## 🚀 Running the Application

### Option 1: Using Streamlit (Recommended)
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Option 2: Using Python
```bash
python -m streamlit run app.py
```

## 📖 How to Use

### 1. **Home Page**
   - Review application overview
   - Understand the purpose and technology
   - Read the disclaimer

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
