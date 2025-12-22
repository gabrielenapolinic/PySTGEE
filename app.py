import streamlit as st
import ee
import geemap.foliumap as geemap
# Import our custom modules
import authentication
import interface

# --- PAGE CONFIG ---
st.set_page_config(page_title="PySTGEE Dashboard", layout="wide")

# --- STEP 1: AUTHENTICATION ---
# If this returns False, it stops the app here and shows the login button
if not authentication.check_gee_auth():
    st.stop()

# --- STEP 2: INTERFACE & CONFIG ---
# This loads the sidebar and gets the user inputs
config = interface.render_sidebar()

# --- STATE MANAGEMENT ---
if 'analysis_active' not in st.session_state:
    st.session_state.analysis_active = False

# Activate analysis if button clicked
if config['start_clicked']:
    st.session_state.analysis_active = True

# --- STEP 3: MAIN LOGIC ---
st.title("PySTGEE: Landslide Hazard Modeling")

if not st.session_state.analysis_active:
    # WAITING STATE
    st.info("👈 System Connected. Configure parameters and click 'Start Analysis'.")

else:
    # RUNNING STATE
    
    # A. MAP
    st.subheader("1. Study Area")
    try:
        m = geemap.Map()
        fc = ee.FeatureCollection(config['polygons_asset'])
        m.centerObject(fc, 10)
        m.addLayer(fc.style(**{'color': 'blue', 'fillColor': '00000000'}), {}, "Slope Units")
        m.to_streamlit(height=450)
    except Exception as e:
        st.error(f"Error loading map: {e}")

    # B. TABS
    st.divider()
    t1, t2, t3 = st.tabs(["📊 Calibration", "⚖️ Validation", "🔮 Prediction"])

    with t1:
        st.write(f"Optimization: {config['min_days']}-{config['max_days']} days")
        if st.button("Run Calibration"):
            st.write("Processing... (Insert Logic Here)")

    with t2:
        if st.button("Run Validation"):
            st.write("Validating... (Insert Logic Here)")

    with t3:
        st.write(f"Target: {config['forecast_date']}")
        if st.button("Run Prediction"):
            st.write("Predicting... (Insert Logic Here)")
