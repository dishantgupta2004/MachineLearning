"""
Energy Efficiency Prediction App
Uses pre-trained Random Forest model
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page config
st.set_page_config(
    page_title="Energy Efficiency Predictor",
    page_icon="🏠",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    model_path = r"D:\\MachineLearning\\EnergyEfficienyMLProject\\notebook\\saved_models\\random_forest_regressor.joblib"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error(f"Model not found at {model_path}")
        return None

model = load_model()

# Title
st.title("🏠 Energy Efficiency Predictor")
st.markdown("Predict **Heating Load** and **Cooling Load** based on building characteristics")
st.markdown("---")

# Input columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Building Geometry")
    
    relative_compactness = st.slider(
        "Relative Compactness",
        min_value=0.62, max_value=0.98, value=0.82, step=0.02
    )
    
    surface_area = st.slider(
        "Surface Area (m²)",
        min_value=514.5, max_value=808.5, value=661.5, step=10.0
    )
    
    wall_area = st.slider(
        "Wall Area (m²)",
        min_value=245.0, max_value=416.5, value=318.5, step=5.0
    )
    
    roof_area = st.slider(
        "Roof Area (m²)",
        min_value=110.25, max_value=220.5, value=147.0, step=5.0
    )

with col2:
    st.subheader("Building Properties")
    
    overall_height = st.select_slider(
        "Overall Height (m)",
        options=[3.5, 7.0], value=7.0
    )
    
    orientation = st.selectbox(
        "Orientation",
        options=[2, 3, 4, 5],
        format_func=lambda x: {2: 'North (2)', 3: 'East (3)', 4: 'South (4)', 5: 'West (5)'}[x]
    )
    
    glazing_area = st.select_slider(
        "Glazing Area",
        options=[0.0, 0.1, 0.25, 0.4], value=0.25
    )
    
    glazing_area_distribution = st.selectbox(
        "Glazing Area Distribution",
        options=[0, 1, 2, 3, 4, 5],
        format_func=lambda x: {0: 'None (0)', 1: 'Uniform (1)', 2: 'North (2)', 
                               3: 'East (3)', 4: 'South (4)', 5: 'West (5)'}[x]
    )

st.markdown("---")

# Predict button
if st.button("🔮 Predict Energy Loads", type="primary", use_container_width=True):
    if model is not None:
        # Prepare input
        input_data = np.array([[
            relative_compactness,
            surface_area,
            wall_area,
            roof_area,
            overall_height,
            orientation,
            glazing_area,
            glazing_area_distribution
        ]])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric(
                label="🔥 Heating Load",
                value=f"{prediction[0][0]:.2f} kWh/m²"
            )
        
        with res_col2:
            st.metric(
                label="❄️ Cooling Load",
                value=f"{prediction[0][1]:.2f} kWh/m²"
            )
        
        # Show input summary
        with st.expander("View Input Parameters"):
            input_df = pd.DataFrame({
                'Parameter': ['Relative Compactness', 'Surface Area', 'Wall Area', 
                             'Roof Area', 'Overall Height', 'Orientation',
                             'Glazing Area', 'Glazing Area Distribution'],
                'Value': [relative_compactness, surface_area, wall_area, roof_area,
                         overall_height, orientation, glazing_area, glazing_area_distribution]
            })
            st.table(input_df)
    else:
        st.error("Model not loaded. Please check the model path.")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app predicts energy efficiency metrics for buildings.
    
    **Model:** Random Forest Regressor  
    **Test R² Score:** 0.9842  
    **Test RMSE:** 1.1973
    
    **Features:**
    - Relative Compactness
    - Surface Area
    - Wall Area
    - Roof Area
    - Overall Height
    - Orientation
    - Glazing Area
    - Glazing Area Distribution
    """)