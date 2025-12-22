import streamlit as st
import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="PySTGEE Config", layout="centered")

st.title("PySTGEE: User Configuration")
st.markdown("Configure the analysis parameters below (Cell 1).")

# --- USER CONFIGURATION FORM ---
with st.form("user_config_form"):
    
    # 1. Earth Engine Project
    st.subheader("1. Earth Engine Project Configuration")
    EE_PROJECT = st.text_input("GEE Project ID", value='stgee-dataset')

    # 2. Asset Paths
    st.subheader("2. Earth Engine Asset Paths")
    polygons_asset = st.text_input("Polygons Asset (Training)", value="projects/stgee-dataset/assets/export_predictors_polygons2")
    points_asset = st.text_input("Points Asset (Events)", value="projects/stgee-dataset/assets/pointsDate")
    prediction_asset = st.text_input("Prediction Asset (Target)", value="projects/stgee-dataset/assets/export_predictors_polygons2")

    # 3. Data Column Settings
    st.subheader("3. Data Column Settings")
    col1, col2 = st.columns(2)
    with col1:
        DATE_COLUMN = st.text_input("Date Column Name", value='formatted_date')
    with col2:
        LANDSLIDE_COLUMN = st.text_input("Landslide ID Column", value='id')

    # 4. CSV Export Settings
    st.subheader("4. CSV Export Settings")
    CSV_EXPORT_MODE = st.selectbox("CSV Export Mode", options=['BEST_ONLY', 'ALL_DATA'], index=0)

    # 5. Analysis Parameters
    st.subheader("5. Analysis Parameters")
    
    # Date Input
    default_date = datetime.date(2025, 11, 26)
    FORECAST_DATE_FIXED = st.date_input("Forecast Date", value=default_date)

    # Predictors (Text Area converted to List)
    default_predictors = "Relief_mea, S_mean, VCv_mean, Hill_mean, NDVI_mean"
    predictors_input = st.text_area("Static Predictors (comma separated)", value=default_predictors)
    
    # Convert string back to list for Python
    STATIC_PREDICTORS = [x.strip() for x in predictors_input.split(',')]

    # Rainfall Range
    st.markdown("**Rainfall Window Search Range (Days)**")
    c1, c2 = st.columns(2)
    with c1:
        MIN_DAYS = st.number_input("Min Days", min_value=1, value=1)
    with c2:
        MAX_DAYS = st.number_input("Max Days", min_value=1, value=30)

    st.markdown("---")
    
    # THE START BUTTON
    submitted = st.form_submit_button("🚀 Start Analysis", type="primary")

# --- ACTION AFTER CLICK ---
if submitted:
    # Save to Session State (Memory)
    st.session_state['config'] = {
        'project': EE_PROJECT,
        'assets': {'poly': polygons_asset, 'pts': points_asset, 'pred': prediction_asset},
        'cols': {'date': DATE_COLUMN, 'ls': LANDSLIDE_COLUMN},
        'csv_mode': CSV_EXPORT_MODE,
        'forecast_date': FORECAST_DATE_FIXED,
        'predictors': STATIC_PREDICTORS,
        'range': (MIN_DAYS, MAX_DAYS)
    }
    
    # Print the confirmation logs as requested
    st.success("Configuration Saved.")
    st.code(f"""
    Project: {EE_PROJECT}
    CSV Mode: {CSV_EXPORT_MODE}
    Rainfall Optimization Range: {MIN_DAYS} to {MAX_DAYS} days.
    Static Predictors: {STATIC_PREDICTORS}
    """)
    
    st.info("System is ready. (Logic for next steps would go here)")
