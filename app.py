import streamlit as st
import ee
import geemap.foliumap as geemap
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_curve, auc
from google.oauth2.credentials import Credentials
import json

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="PySTGEE Dashboard", layout="wide")

st.title("PySTGEE: Landslide Hazard Modeling")
st.markdown("""
**Interactive Dashboard for GEE Landslide Forecasting.** 1. Configure data in the sidebar.
2. Click **Run Analysis** to load the study area.
3. Use the tabs to Calibrate, Validate, and Predict.
""")

# --- 2. AUTHENTICATION ---
def check_gee_auth():
    DEFAULT_PROJECT = 'stgee-dataset' 

    # 1. Try Secrets (Cloud)
    if "EARTHENGINE_TOKEN" in st.secrets:
        try:
            token_data = st.secrets["EARTHENGINE_TOKEN"]
            if isinstance(token_data, str):
                try:
                    token_dict = json.loads(token_data)
                    creds = Credentials.from_authorized_user_info(token_dict)
                except:
                    st.error("Invalid Secret Format.")
                    st.stop()
            else:
                creds = Credentials.from_authorized_user_info(dict(token_data))
            
            ee.Initialize(credentials=creds, project=DEFAULT_PROJECT)
            return True
        except Exception:
            pass

    # 2. Try Local
    try:
        ee.Initialize(project=DEFAULT_PROJECT)
        return True
    except Exception:
        st.warning("⚠️ Google Earth Engine access not detected.")
        if st.button("🔐 Authenticate via Browser"):
            ee.Authenticate()
            ee.Initialize(project=DEFAULT_PROJECT)
            st.rerun()
        st.stop()

check_gee_auth()

# --- 3. SESSION STATE INIT ---
if 'analysis_active' not in st.session_state:
    st.session_state.analysis_active = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'best_days' not in st.session_state:
    st.session_state.best_days = 1
if 'final_predictors' not in st.session_state:
    st.session_state.final_predictors = []
if 'training_df' not in st.session_state:
    st.session_state.training_df = None

# --- 4. SIDEBAR ---
st.sidebar.header("1. Data Assets")
polygons_asset = st.sidebar.text_input("Polygons Asset (Training)", "projects/stgee-dataset/assets/export_predictors_polygons2")
points_asset = st.sidebar.text_input("Points Asset (Events)", "projects/stgee-dataset/assets/pointsDate")
prediction_asset = st.sidebar.text_input("Prediction Asset (Target)", "projects/stgee-dataset/assets/export_predictors_polygons2")

DATE_COLUMN = st.sidebar.text_input("Date Column Name", "formatted_date")
LANDSLIDE_COLUMN = st.sidebar.text_input("Landslide ID Column", "id")

st.sidebar.header("2. Model Parameters")
MIN_DAYS = st.sidebar.number_input("Min Rainfall Window (Days)", 1, 60, 1)
MAX_DAYS = st.sidebar.number_input("Max Rainfall Window (Days)", 1, 60, 30)
FORECAST_DATE = st.sidebar.date_input("Forecast Date", pd.to_datetime("2025-11-26"))
STATIC_PREDICTORS = ['Relief_mea', 'S_mean', 'VCv_mean', 'Hill_mean', 'NDVI_mean']

# --- 5. LOGIC FUNCTIONS ---
@st.cache_data(ttl=3600, show_spinner=False)
def download_training_data(min_d, max_d, date_col, _polygons_path, _points_path):
    poly_fc = ee.FeatureCollection(_polygons_path)
    points_fc = ee.FeatureCollection(_points_path)
    raw_dates = points_fc.aggregate_array(date_col).distinct().getInfo()
    dates_list = [str(d)[:10] for d in raw_dates]
    results = []
    
    progress_text = "Downloading training data..."
    my_bar = st.progress(0, text=progress_text)
    total = len(dates_list)
    
    for idx, date_str in enumerate(dates_list):
        try:
            d = ee.Date(date_str)
            gpm = ee.ImageCollection('JAXA/GPM_L3/GSMaP/v8/operational').select('hourlyPrecipRateGC')
            rain_bands = [gpm.filterDate(d.advance(-i, 'day'), d).sum().unmask(0).rename(f'Rn{i}') for i in range(min_d, max_d + 1)]
            combined = ee.Image.cat(rain_bands)
            todays_points = points_fc.filter(ee.Filter.eq(date_col, date_str))
            
            def map_polygons(poly):
                count = todays_points.filterBounds(poly.geometry()).size()
                return poly.set({'P/A': ee.Algorithms.If(count.gt(0), 1, 0), 'date': date_str})
            
            labeled_polys = poly_fc.map(map_polygons)
            
            def add_numeric_id(feature):
                str_id = ee.String(feature.get('id'))
                num_str = str_id.replace(r'[^0-9]', '', 'g')
                num_val = ee.Algorithms.If(num_str.length().gt(0), ee.Number.parse(num_str), 0)
                return feature.set('NUM_ID', num_val)
                
            labeled_polys = labeled_polys.map(add_numeric_id)
            stats = combined.reduceRegions(collection=labeled_polys, reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True), scale=1000, tileScale=16)
            df_day = geemap.ee_to_df(stats)
            if not df_day.empty:
                rename_dict = {f'Rn{i}_{suffix}': f'Rn{i}_{m}' for i in range(min_d, max_d + 1) for suffix, m in [('mean', 'm'), ('stdDev', 's')]}
                df_day = df_day.rename(columns=rename_dict)
                results.append(df_day)
        except: pass
        my_bar.progress((idx + 1) / total, text=f"Processing {date_str}...")
        
    my_bar.empty()
    if not results: return pd.DataFrame()
    final_df = pd.concat(results, ignore_index=True)
    if 'date' in final_df.columns and 'id' in final_df.columns:
        final_df = final_df.sort_values(by=['date', 'id']).reset_index(drop=True)
    for col in STATIC_PREDICTORS:
        if col not in final_df.columns: final_df[col] = 0
    return final_df.fillna(0)

def visualize_map(df, value_col, layer_name, asset_path, palette):
    df_map = df.copy()
    df_map['NUM_ID_PY'] = df_map['id'].apply(lambda x: int(''.join(filter(str.isdigit, str(x))) or 0))
    df_flat = df_map.groupby('NUM_ID_PY')[value_col].max().reset_index()
    ids = df_flat['NUM_ID_PY'].tolist()
    vals = df_flat[value_col].tolist()
    
    fc = ee.FeatureCollection(asset_path)
    def add_id(f):
        str_id = ee.String(f.get('id'))
        num = ee.Number.parse(str_id.replace(r'[^0-9]', '', 'g'))
        return f.set('NUM_ID', num)
    
    fc_mapped = fc.map(add_id)
    fc_img = fc_mapped.reduceToImage(['NUM_ID'], ee.Reducer.first())
    result_img = fc_img.remap(ids, vals).rename('val')
    result_img = result_img.updateMask(result_img.gte(0))
    
    m = geemap.Map()
    m.centerObject(fc, 10)
    vis_params = {'min': 0, 'max': 1, 'palette': palette}
    if "Confusion" in layer_name:
         vis_params = {'min': 0, 'max': 3, 'palette': ['#D10E00', '#DF564D', '#DBEADD', '#006B0B']}
    m.addLayer(result_img, vis_params, layer_name)
    return m

# --- 6. MAIN FLOW ---

# BLOCK 1: START BUTTON (This is what you see first)
if not st.session_state.analysis_active:
    st.info("Click the button below to initialize the study area map and enable analysis tools.")
    if st.button("🚀 Run Analysis", type="primary"):
        st.session_state.analysis_active = True
        st.rerun()

# BLOCK 2: ANALYSIS DASHBOARD (Visible only after clicking Run Analysis)
else:
    # --- BASE MAP (Visible IMMEDIATELY) ---
    st.markdown("### 🗺️ Study Area Map")
    with st.expander("Show/Hide Base Map", expanded=True):
        # Create a base map showing just the polygons initially
        m_base = geemap.Map()
        try:
            fc_polys = ee.FeatureCollection(polygons_asset)
            m_base.centerObject(fc_polys, 10)
            # Add polygons as a transparent layer with outlines
            m_base.addLayer(fc_polys.style(**{'color': 'blue', 'fillColor': '00000000'}), {}, "Study Area Polygons")
        except Exception as e:
            st.error(f"Error loading polygons: {e}")
        
        m_base.to_streamlit(height=500)

    st.markdown("---")

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["🚀 Calibration", "⚖️ Validation", "🔮 Prediction"])

    with tab1:
        st.subheader("Model Calibration")
        st.write("Click below to download rainfall data and train the Random Forest model.")
        
        if st.button("Start Calibration Process", type="primary"):
            with st.spinner("Processing... This may take a minute."):
                df = download_training_data(MIN_DAYS, MAX_DAYS, DATE_COLUMN, polygons_asset, points_asset)
                
                if not df.empty:
                    st.session_state.training_df = df
                    y = df['P/A']
                    best_auc = 0
                    best_w = MIN_DAYS
                    
                    # Optimization
                    for days in range(MIN_DAYS, MAX_DAYS + 1):
                        cols = [f'Rn{days}_m', f'Rn{days}_s']
                        if all(c in df.columns for c in cols):
                            X_tmp = df[cols].fillna(0)
                            rf = RandomForestClassifier(n_estimators=30, max_depth=7, class_weight='balanced', random_state=42)
                            rf.fit(X_tmp, y)
                            probs = rf.predict_proba(X_tmp)[:, 1]
                            score = auc(*roc_curve(y, probs)[:2])
                            if score > best_auc:
                                best_auc = score
                                best_w = days
                    
                    st.session_state.best_days = best_w
                    st.session_state.final_predictors = STATIC_PREDICTORS + [f'Rn{best_w}_m', f'Rn{best_w}_s']
                    
                    # Final Fit
                    X = df[st.session_state.final_predictors].fillna(0)
                    rf_final = RandomForestClassifier(n_estimators=100, max_depth=10, oob_score=True, class_weight='balanced', random_state=42)
                    rf_final.fit(X, y)
                    st.session_state.model = rf_final
                    
                    st.success(f"Calibration Done! Best Window: {best_w} Days (AUC: {best_auc:.3f})")
                    
                    # Result Map
                    probs = rf_final.oob_decision_function_[:, 1]
                    df['calib_prob'] = probs
                    st.markdown("#### Calibration Results Map")
                    m_calib = visualize_map(df, 'calib_prob', 'Calibration Probability', polygons_asset, ['white', 'green', 'yellow', 'red'])
                    m_calib.to_streamlit(height=500)
                else:
                    st.error("No data found.")

    with tab2:
        st.subheader("Validation")
        if st.button("Run Validation"):
            if st.session_state.model:
                df = st.session_state.training_df
                X = df[st.session_state.final_predictors].fillna(0)
                y = df['P/A']
                y_probs = cross_val_predict(st.session_state.model, X, y, cv=StratifiedKFold(10), method='predict_proba')[:, 1]
                cv_auc = auc(*roc_curve(y, y_probs)[:2])
                st.metric("Validation AUC", f"{cv_auc:.4f}")
                
                df['valid_prob'] = y_probs
                m_valid = visualize_map(df, 'valid_prob', 'Validation Probability', polygons_asset, ['white', 'orange', 'red'])
                m_valid.to_streamlit(height=500)
            else:
                st.warning("Train model first.")

    with tab3:
        st.subheader("Prediction")
        if st.button("Run Prediction"):
            if st.session_state.model:
                target_d = ee.Date(FORECAST_DATE.strftime('%Y-%m-%d'))
                days = st.session_state.best_days
                gpm = ee.ImageCollection('JAXA/GPM_L3/GSMaP/v8/operational').select('hourlyPrecipRateGC')
                latest = gpm.filterDate('2000-01-01', target_d.advance(1, 'day')).sort('system:time_start', False).first()
                
                if latest:
                    found_date = ee.Date(latest.get('system:time_start'))
                    rain_img = gpm.filterDate(found_date.advance(-days, 'day'), found_date.advance(1, 'day')).sum().unmask(0)
                    
                    pred_fc = ee.FeatureCollection(prediction_asset)
                    def add_num_id(f):
                        s = ee.String(f.get('id'))
                        n = ee.Number.parse(s.replace(r'[^0-9]', '', 'g'))
                        return f.set('NUM_ID', n)
                    pred_fc = pred_fc.map(add_num_id)
                    
                    stats = rain_img.reduceRegions(collection=pred_fc, reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True), scale=1000)
                    df_pred = geemap.ee_to_df(stats)
                    
                    col_mean = f'Rn{days}_m'
                    col_std = f'Rn{days}_s'
                    m_c = [c for c in df_pred.columns if 'mean' in c]
                    s_c = [c for c in df_pred.columns if 'stdDev' in c]
                    df_pred[col_mean] = df_pred[m_c[0]] if m_c else 0
                    df_pred[col_std] = df_pred[s_c[0]] if s_c else 0
                    
                    for col in STATIC_PREDICTORS:
                        if col not in df_pred.columns: df_pred[col] = 0
                        
                    X_pred = df_pred[st.session_state.final_predictors].fillna(0)
                    probs = st.session_state.model.predict_proba(X_pred)[:, 1]
                    df_pred['SI'] = probs
                    
                    st.success(f"Max Risk: {probs.max():.2f}")
                    m_pred = visualize_map(df_pred, 'SI', 'Landslide Susceptibility', prediction_asset, ['green', 'yellow', 'red'])
                    m_pred.to_streamlit(height=600)
                    
                    csv = df_pred.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Prediction CSV", csv, "prediction.csv", "text/csv")
            else:
                st.warning("Train model first.")
