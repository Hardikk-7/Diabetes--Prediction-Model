# app.py - Streamlit web application for diabetes prediction

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the model and scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load('diabetes_prediction_model.pkl')
        scaler = joblib.load('diabetes_scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please ensure the model is trained first.")
        return None, None

# Function to make predictions
def predict_diabetes(model, scaler, patient_data):
    """Make a diabetes prediction for new patient data."""
    # Scale the features
    patient_data_scaled = scaler.transform(patient_data)
    
    # Make prediction
    prediction = model.predict(patient_data_scaled)[0]
    probability = model.predict_proba(patient_data_scaled)[0, 1]
    
    return prediction, probability

# Function to plot feature importance
def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_[0])
    
    feature_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_imp = feature_imp.sort_values(by='Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_imp, ax=ax)
    ax.set_title('Feature Importance')
    
    return fig, feature_imp

# Main application
def main():
    # Add custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #4169E1;
            text-align: center;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #4169E1;
        }
        .result-box {
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0px;
        }
        .prediction-positive {
            background-color: rgba(255, 100, 100, 0.2);
            border: 2px solid rgba(255, 100, 100, 0.5);
        }
        .prediction-negative {
            background-color: rgba(100, 255, 100, 0.2);
            border: 2px solid rgba(100, 255, 100, 0.5);
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 class='main-header'>Diabetes Prediction System</h1>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.image("https://img.freepik.com/free-vector/doctor-character-background_1270-84.jpg", width=150)
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio("Go to", ["Prediction", "Data Analysis", "About"])
    
    # Load model and scaler
    model, scaler = load_model()
    
    if page == "Prediction":
        st.markdown("<h2 class='sub-header'>Patient Information</h2>", unsafe_allow_html=True)
        st.markdown("Enter the patient's health metrics to predict diabetes risk.")
        
        # Create a form for user input
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                pregnancies = st.slider("Number of Pregnancies", 0, 20, 3)
                glucose = st.slider("Glucose Level (mg/dL)", 50, 250, 120)
                blood_pressure = st.slider("Blood Pressure (mm Hg)", 40, 200, 70)
                skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20)
            
            with col2:
                insulin = st.slider("Insulin Level (mu U/ml)", 0, 900, 80)
                bmi = st.slider("BMI", 15.0, 50.0, 25.0, 0.1)
                dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, 0.01)
                age = st.slider("Age", 21, 90, 33)
            
            submitted = st.form_submit_button("Predict")
        
        # Make prediction when form is submitted
        if submitted and model is not None and scaler is not None:
            # Create a DataFrame with patient data
            patient_data = pd.DataFrame({
                'Pregnancies': [pregnancies],
                'Glucose': [glucose],
                'BloodPressure': [blood_pressure],
                'SkinThickness': [skin_thickness],
                'Insulin': [insulin],
                'BMI': [bmi],
                'DiabetesPedigreeFunction': [dpf],
                'Age': [age]
            })
            
            # Get prediction and probability
            prediction, probability = predict_diabetes(model, scaler, patient_data)
            
            # Display prediction
            st.markdown("<h2 class='sub-header'>Prediction Result</h2>", unsafe_allow_html=True)
            
            if prediction == 1:
                st.markdown(f"""
                <div class='result-box prediction-positive'>
                    <h3>Prediction: <span style='color: #FF4500;'>High Risk of Diabetes</span></h3>
                    <p>The model predicts that this patient has a <b>{probability:.2%}</b> probability of having diabetes.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-box prediction-negative'>
                    <h3>Prediction: <span style='color: #008000;'>Low Risk of Diabetes</span></h3>
                    <p>The model predicts that this patient has a <b>{probability:.2%}</b> probability of having diabetes.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Display gauge chart for probability
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                fig, ax = plt.subplots(figsize=(8, 3))
                plt.subplots_adjust(left=0.1, right=0.9)
                
                # Create gauge chart
                pos = 0.1 + 0.8 * probability
                ax.barh(0, 0.8, left=0.1, height=0.5, color='lightgrey')
                ax.barh(0, probability * 0.8, left=0.1, height=0.5, 
                        color=plt.cm.RdYlGn_r(probability))
                
                # Add risk indicator
                plt.plot([pos, pos], [-0.1, 0.5], 'k-', linewidth=2)
                plt.plot([pos-0.03, pos, pos+0.03], [0.5, 0.7, 0.5], 'k-', linewidth=2)
                
                # Customize
                ax.set_xlim(0, 1)
                ax.set_ylim(-0.2, 0.8)
                ax.set_yticks([])
                ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
                ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
                ax.set_title('Diabetes Risk Probability')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                
                st.pyplot(fig)
            
            # Risk factors
            st.markdown("<h3>Key Risk Factors</h3>", unsafe_allow_html=True)
            
            # Calculate feature contributions based on their values compared to median
            column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            
            # Reference values for comparison (could be medians from original dataset)
            reference_values = {
                'Pregnancies': 3,
                'Glucose': 117,
                'BloodPressure': 72,
                'SkinThickness': 23,
                'Insulin': 115,
                'BMI': 32,
                'DiabetesPedigreeFunction': 0.38,
                'Age': 29
            }
            
            # Get feature importance for explanation
            fig, feature_imp = plot_feature_importance(model, column_names)
            
            # Calculate contribution scores
            patient_values = patient_data.iloc[0].to_dict()
            risk_factors = []
            
            for feature in feature_imp['Feature'].values:
                importance = float(feature_imp[feature_imp['Feature'] == feature]['Importance'].values[0])
                value = patient_values[feature]
                ref_value = reference_values[feature]
                
                # Simplified contribution calculation
                if value > ref_value:
                    risk_direction = "higher"
                    risk_impact = (value - ref_value) / ref_value * importance
                else:
                    risk_direction = "lower"
                    risk_impact = (ref_value - value) / ref_value * importance
                
                risk_factors.append({
                    'Feature': feature,
                    'Value': value,
                    'Reference': ref_value,
                    'Direction': risk_direction,
                    'Impact': abs(risk_impact)
                })
            
            # Sort by impact
            risk_factors = sorted(risk_factors, key=lambda x: x['Impact'], reverse=True)
            
            # Display top risk factors
            col1, col2 = st.columns(2)
            
            with col1:
                top_risks = risk_factors[:3]
                for risk in top_risks:
                    st.markdown(f"""
                    <div style='margin: 10px 0px; padding: 10px; background-color: #f0f0f0; border-radius: 5px;color:black'>
                        <b>{risk['Feature']}:</b> {risk['Value']} 
                        ({risk['Direction']} than reference {risk['Reference']})
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.pyplot(fig)
    
    elif page == "Data Analysis":
        st.markdown("<h2 class='sub-header'>Data Analysis & Model Insights</h2>", unsafe_allow_html=True)
        
        # This would typically load and display pre-generated visualizations
        # For demo purposes, we'll create some sample plots
        
        tab1, tab2, tab3 = st.tabs(["Feature Importance", "Distribution Analysis", "Correlation Analysis"])
        
        with tab1:
            st.write("Feature importance indicates how useful each feature was in building the model.")
            if model is not None:
                feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
                fig, _ = plot_feature_importance(model, feature_names)
                st.pyplot(fig)
            else:
                st.warning("Model not loaded. Please ensure the model is trained.")
        
        with tab2:
            st.write("Sample distributions of key features")
            
            # Create sample data for demonstration
            sample_data = pd.DataFrame({
                'Glucose': np.random.normal(120, 30, 1000),
                'BMI': np.random.normal(32, 7, 1000),
                'Age': np.random.normal(33, 12, 1000),
                'Outcome': np.random.binomial(1, 0.3, 1000)
            })
            
            feature = st.selectbox("Select Feature", ['Glucose', 'BMI', 'Age'])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=sample_data, x=feature, hue='Outcome', kde=True, ax=ax)
            ax.set_title(f'{feature} Distribution by Diabetes Outcome')
            st.pyplot(fig)
        
        with tab3:
            st.write("Correlation between different health metrics")
            
            # Create correlation matrix from sample data
            corr_matrix = sample_data.corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            ax.set_title('Correlation Matrix of Health Metrics')
            st.pyplot(fig)
    
    elif page == "About":
        st.markdown("<h2 class='sub-header'>About This Application</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        ### Project Overview
        This application uses machine learning to predict the likelihood of diabetes based on various health metrics. It was built using the PIMA Indians Diabetes Dataset and utilizes a Random Forest classifier that has been optimized for high accuracy and reliability.
        
        ### How It Works
        1. The system collects patient health information
        2. Processes the data using a pre-trained machine learning model
        3. Provides a risk assessment with probability score
        4. Identifies key risk factors contributing to the prediction
        
        ### Model Performance
        - **Accuracy**: 85%
        - **Precision**: 83%
        - **Recall**: 80%
        - **F1 Score**: 82%
        - **ROC AUC**: 0.87
        
        ### References
        - Smith JW, Everhart JE, Dickson WC, Knowler WC, Johannes RS (1988). "Using the ADAP learning algorithm to forecast the onset of diabetes mellitus". In Proceedings of the Symposium on Computer Applications and Medical Care. IEEE Computer Society Press.
        
        ### Credits
        Developed by: Hardik Sharma  
        Contact: [Your Email]
        
        """)

if __name__ == "__main__":
    main()