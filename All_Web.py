# -*- coding: utf-8 -*-
"""
Diabetes Prediction Web Application

A Streamlit-based application for predicting diabetes risk using
machine learning models trained on historical medical data.

@author: Machine Learning Team
"""

import pickle
import numpy as np
import streamlit as st
import warnings
warnings.filterwarnings('ignore')


# Configure Streamlit page settings
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load pre-trained model and scaler
@st.cache_resource
def load_models():
    """Load the trained diabetes model and scaler from saved files."""
    try:
        with open("trained_diabetes_model.sav", 'rb') as model_file:
            model = pickle.load(model_file)
        with open("scaler.sav", 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Error: Could not load model files. {e}")
        st.stop()

diabetes_model, diabetes_scaler = load_models()


def predict_diabetes(input_data):
    """
    Predict diabetes risk based on input medical parameters.
    
    Args:
        input_data: List of 8 medical parameters
        
    Returns:
        str: Prediction result with interpretation
    """
    try:
        # Convert input to numpy array
        input_array = np.asarray(input_data, dtype=float)
        
        # Reshape for single prediction
        input_reshaped = input_array.reshape(1, -1)
        
        # Scale the input data
        scaled_data = diabetes_scaler.transform(input_reshaped)
        
        # Make prediction
        prediction = diabetes_model.predict(scaled_data)
        
        if prediction[0] == 0:
            return "Negative", "The person is likely NOT diabetic"
        else:
            return "Positive", "The person is likely diabetic"
            
    except ValueError as e:
        return "Error", f"Invalid input: Please ensure all fields contain valid numbers. {str(e)}"
    except Exception as e:
        return "Error", f"Prediction error: {str(e)}"


def main():
    """Main application function."""
    # Add header
    st.title("🏥 Diabetes Prediction System")
    st.markdown(
        """
        This application uses a machine learning model to predict the likelihood of diabetes
        based on medical parameters. Please enter your medical information below.
        """
    )
    
    # Add info box
    with st.info():
        st.markdown("""
        **Note:** This prediction tool is for educational purposes only and should not be 
        used as a substitute for professional medical advice. Please consult a healthcare 
        provider for accurate diagnosis and treatment.
        """)
    
    # Create form for input
    st.subheader("Medical Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input(
            "Number of Pregnancies",
            min_value=0,
            max_value=17,
            value=0,
            step=1,
            help="Number of times pregnant"
        )
        glucose = st.number_input(
            "Glucose Level (mg/dL)",
            min_value=0.0,
            max_value=300.0,
            value=100.0,
            step=0.1,
            help="Plasma glucose concentration (mg/dL)"
        )
        blood_pressure = st.number_input(
            "Blood Pressure (mm Hg)",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=0.1,
            help="Diastolic blood pressure (mm Hg)"
        )
        skin_thickness = st.number_input(
            "Skin Thickness (mm)",
            min_value=0.0,
            max_value=100.0,
            value=20.0,
            step=0.1,
            help="Triceps skin fold thickness (mm)"
        )
    
    with col2:
        insulin = st.number_input(
            "Insulin Level (IU/mL)",
            min_value=0.0,
            max_value=900.0,
            value=80.0,
            step=0.1,
            help="2-hour serum insulin (IU/mL)"
        )
        bmi = st.number_input(
            "BMI (Body Mass Index)",
            min_value=0.0,
            max_value=70.0,
            value=25.0,
            step=0.1,
            help="Body mass index (weight in kg/(height in m)^2)"
        )
        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5,
            step=0.01,
            help="Genetic predisposition to diabetes"
        )
        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=150,
            value=30,
            step=1,
            help="Age of the person in years"
        )
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Predict Diabetes Risk", use_container_width=True):
            # Prepare input data
            input_data = [
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                diabetes_pedigree,
                age
            ]
            
            # Make prediction
            result_status, result_message = predict_diabetes(input_data)
            
            # Display results
            st.divider()
            st.subheader("Prediction Result")
            
            if result_status == "Positive":
                st.warning(f"⚠️ {result_message}")
                st.markdown("""
                **Recommendations:**
                - Schedule an appointment with an endocrinologist
                - Monitor your blood glucose levels regularly
                - Maintain a healthy diet and exercise routine
                - Reduce stress and get adequate sleep
                """)
            elif result_status == "Negative":
                st.success(f"✅ {result_message}")
                st.markdown("""
                **Recommendations:**
                - Continue regular health checkups
                - Maintain a healthy lifestyle
                - Stay physically active
                - Monitor your health regularly
                """)
            else:
                st.error(f"❌ {result_message}")
    
    # Footer
    st.divider()
    st.markdown(
        """
        ---
        **Disclaimer:** This tool provides predictions based on a machine learning model 
        and should not be considered medical advice. Always consult qualified healthcare 
        professionals for diagnosis and treatment decisions.
        """
    )


if __name__ == "__main__":
    main()
        
        
        
    
    
    
    
    
    
    
    
    
    
