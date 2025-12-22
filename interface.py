import streamlit as st
import datetime

def render_sidebar():
    """
    Renders the configuration sidebar and returns inputs.
    """
    st.sidebar.title("PySTGEE Config")
    
    # GEE Project
    st.sidebar.subheader("Earth Engine Project")
    ee_project = st.sidebar.text_input("GEE Project ID", value='stgee-dataset')

    # Assets
    st.sidebar.subheader("Asset Paths")
    poly_asset = st.sidebar.text_input("Polygons Asset", value="projects/stgee-dataset/assets/export_predictors_polygons2")
    pts_asset = st.sidebar.text_input("Points Asset", value="projects/stgee-dataset/assets/pointsDate")
    pred_asset = st.sidebar.text_input("Prediction Asset", value="projects/stgee-dataset/assets/export_predictors_polygons2")

    # Columns
    st.sidebar.subheader("Column Settings")
    date_col = st.sidebar.text_input("Date Column", value='formatted_date')
    ls_col = st.sidebar.text_input("Landslide ID", value='id')

    # Params
    st.sidebar.subheader("Parameters")
    fc_date = st.sidebar.date_input("Forecast Date", value=datetime.date(2025, 11, 26))
    
    # Rainfall
    st.sidebar.caption("Rainfall Window (Days)")
    c1, c2 = st.sidebar.columns(2)
    min_d = c1.number_input("Min", 1, 60, 1)
    max_d = c2.number_input("Max", 1, 60, 30)

    st.sidebar.markdown("---")
    
    # Start Button
    start_btn = st.sidebar.button("🚀 Start Analysis", type="primary")

    return {
        'project': ee_project,
        'polygons_asset': poly_asset,
        'points_asset': pts_asset,
        'prediction_asset': pred_asset,
        'date_col': date_col,
        'forecast_date': fc_date,
        'min_days': min_d,
        'max_days': max_d,
        'start_clicked': start_btn
    }
