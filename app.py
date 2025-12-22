import streamlit as st
import ee
import geemap.foliumap as geemap
import interface  # IMPORTIAMO IL FILE CREATO SOPRA

# --- PAGE CONFIG ---
st.set_page_config(page_title="PySTGEE Dashboard", layout="wide")

# --- AUTHENTICATION (SILENT) ---
# Assumes 'earthengine authenticate' was run in terminal
try:
    ee.Initialize(project='stgee-dataset') # Default fallback
except Exception:
    st.warning("⚠️ GEE not initialized. Please run authentication in terminal.")

# --- STATE MANAGEMENT ---
if 'analysis_active' not in st.session_state:
    st.session_state.analysis_active = False

# --- 1. RENDER SIDEBAR (FROM INTERFACE.PY) ---
# This draws the inputs on the left and gets the values
config = interface.render_sidebar()

# Check if button was clicked
if config['start_clicked']:
    st.session_state.analysis_active = True
    # Optional: Re-initialize with the specific project user entered
    try:
        ee.Initialize(project=config['project'])
        st.toast(f"Connected to project: {config['project']}")
    except Exception as e:
        st.error(f"Connection failed: {e}")

# --- 2. MAIN CONTENT AREA ---
st.title("PySTGEE: Landslide Hazard Modeling")

if not st.session_state.analysis_active:
    # STATE: WAITING FOR START
    st.info("👈 Please configure the parameters in the sidebar and click 'Start Analysis'.")

else:
    # STATE: ANALYSIS STARTED
    
    # A. SHOW MAP WITH FEATURE COLLECTION
    st.subheader("1. Study Area Visualization")
    try:
        m = geemap.Map()
        # Load the polygons from the config
        fc = ee.FeatureCollection(config['polygons_asset'])
        
        # Center and Add Layer
        m.centerObject(fc, 10)
        m.addLayer(fc.style(**{'color': 'blue', 'fillColor': '00000000'}), {}, "Slope Units")
        
        # Display Map
        m.to_streamlit(height=500)
    except Exception as e:
        st.error(f"Error loading map assets: {e}")

    st.divider()

    # B. SHOW TABS (Calibration, Validation, Prediction)
    st.subheader("2. Modeling Pipeline")
    
    tab1, tab2, tab3 = st.tabs(["📊 Calibration", "⚖️ Validation", "🔮 Prediction"])

    with tab1:
        st.markdown("#### Model Calibration")
        st.write(f"Optimization Range: **{config['min_days']} - {config['max_days']} days**")
        if st.button("Run Calibration"):
            st.write("🔄 Calibration process started... (Logic to be implemented)")
            # Qui inseriresti la logica di calibrazione usando config['polygons_asset'], ecc.

    with tab2:
        st.markdown("#### Cross-Validation")
        if st.button("Run Validation"):
            st.write("🔄 Validation process started... (Logic to be implemented)")

    with tab3:
        st.markdown("#### Prediction")
        st.write(f"Target Date: **{config['forecast_date']}**")
        if st.button("Run Prediction"):
            st.write("🔄 Prediction process started... (Logic to be implemented)")
