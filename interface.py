import streamlit as st
import datetime

def render_sidebar():
    """
    Renders the configuration sidebar and returns the user inputs as a dictionary.
    """
    st.sidebar.title("PySTGEE Config")
    
    # --- SECTION: Earth Engine Project ---
    st.sidebar.subheader("Earth Engine Project")
    ee_project = st.sidebar.text_input("GEE Project ID", value='stgee-dataset')

    # --- SECTION: Asset Paths ---
    st.sidebar.subheader("Asset Paths")
    poly_asset = st.sidebar.text_input("Polygons Asset (Training)", value="projects/stgee-dataset/assets/export_predictors_polygons2")
    pts_asset = st.sidebar.text_input("Points Asset (Events)", value="projects/stgee-dataset/assets/pointsDate")
    pred_asset = st.sidebar.text_input("Prediction Asset (Target)", value="projects/stgee-dataset/assets/export_predictors_polygons2")

    # --- SECTION: Data Columns ---
    st.sidebar.subheader("Column Settings")
    date_col = st.sidebar.text_input("Date Column Name", value='formatted_date')
    ls_col = st.sidebar.text_input("Landslide ID Column", value='id')

    # --- SECTION: Export Settings ---
    st.sidebar.subheader("Export Settings")
    csv_mode = st.sidebar.selectbox("CSV Mode", options=['BEST_ONLY', 'ALL_DATA'], index=0)

    # --- SECTION: Analysis Parameters ---
    st.sidebar.subheader("Analysis Parameters")
    
    # Forecast Date
    default_date = datetime.date(2025, 11, 26)
    fc_date = st.sidebar.date_input("Forecast Date", value=default_date)

    # Predictors
    default_preds = "Relief_mea, S_mean, VCv_mean, Hill_mean, NDVI_mean"
    preds_input = st.sidebar.text_area("Static Predictors", value=default_preds, help="Comma separated values")
    static_preds = [x.strip() for x in preds_input.split(',')]

    # Rainfall Range
    st.sidebar.caption("Rainfall Window Range (Days)")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        min_d = st.number_input("Min", min_value=1, value=1)
    with c2:
        max_d = st.number_input("Max", min_value=1, value=30)

    st.sidebar.markdown("---")
    
    # THE START BUTTON
    # We return the button state directly
    start_btn = st.sidebar.button("🚀 Start Analysis", type="primary")

    # Pack everything into a dictionary
    config = {
        'project': ee_project,
        'polygons_asset': poly_asset,
        'points_asset': pts_asset,
        'prediction_asset': pred_asset,
        'date_col': date_col,
        'landslide_col': ls_col,
        'csv_mode': csv_mode,
        'forecast_date': fc_date,
        'static_predictors': static_preds,
        'min_days': min_d,
        'max_days': max_d,
        'start_clicked': start_btn
    }
    
    return config
