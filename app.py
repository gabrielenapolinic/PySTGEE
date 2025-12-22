import streamlit as st
import ee
import geemap.foliumap as geemap
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="PySTGEE Dashboard", layout="wide")

st.title("PySTGEE: Landslide Hazard Modeling")
st.markdown("""
**Interactive Dashboard for GEE Landslide Forecasting.** Adjust parameters in the sidebar, then run Calibration, Validation, or Prediction in the tabs below.
""")

# --- 2. GEE AUTHENTICATION ---
# This function handles authentication safely for both local and cloud usage
def initialize_gee():
    try:
        # Try standard initialization (works if you have local credentials)
        ee.Initialize(project='stgee-dataset')
        return True
    except Exception as e:
        # If running on Streamlit Cloud, you might need secrets
        # See instructions below the code on how to set this up
        try:
            # Placeholder for Cloud Authentication logic
            # For local use, 'earthengine authenticate' in terminal is enough
            st.error(f"GEE Initialization failed. Please run 'earthengine authenticate' locally. Error: {e}")
            return False
        except Exception:
            return False

if not initialize_gee():
    st.stop()

# --- 3. SIDEBAR CONFIGURATION ---
st.sidebar.header("1. Data Configuration")

# Using session state to persist these inputs
polygons_asset = st.sidebar.text_input("Polygons Asset", "projects/stgee-dataset/assets/export_predictors_polygons2")
points_asset = st.sidebar.text_input("Points Asset", "projects/stgee-dataset/assets/pointsDate")
prediction_asset = st.sidebar.text_input("Prediction Asset", "projects/stgee-dataset/assets/export_predictors_polygons2")

DATE_COLUMN = st.sidebar.text_input("Date Column Name", "formatted_date")
LANDSLIDE_COLUMN = st.sidebar.text_input("Landslide ID Column", "id")

st.sidebar.header("2. Model Parameters")
MIN_DAYS = st.sidebar.number_input("Min Rainfall Window (Days)", 1, 60, 1)
MAX_DAYS = st.sidebar.number_input("Max Rainfall Window (Days)", 1, 60, 30)
FORECAST_DATE = st.sidebar.date_input("Forecast Date", pd.to_datetime("2025-11-26"))

STATIC_PREDICTORS = ['Relief_mea', 'S_mean', 'VCv_mean', 'Hill_mean', 'NDVI_mean']

# Initialize Session State variables
if 'model' not in st.session_state:
    st.session_state.model = None
if 'best_days' not in st.session_state:
    st.session_state.best_days = MIN_DAYS
if 'final_predictors' not in st.session_state:
    st.session_state.final_predictors = []
if 'training_df' not in st.session_state:
    st.session_state.training_df = None

# --- 4. CORE LOGIC FUNCTIONS ---

@st.cache_data(ttl=3600, show_spinner=False)
def download_training_data(min_d, max_d, date_col, _polygons_path, _points_path):
    """
    Downloads and processes training data from GEE.
    Cached to prevent re-downloading on every interaction.
    """
    poly_fc = ee.FeatureCollection(_polygons_path)
    points_fc = ee.FeatureCollection(_points_path)
    
    # Get unique dates
    raw_dates = points_fc.aggregate_array(date_col).distinct().getInfo()
    dates_list = [str(d)[:10] for d in raw_dates]
    
    results = []
    
    # Progress bar logic
    progress_text = "Downloading data from Earth Engine..."
    my_bar = st.progress(0, text=progress_text)
    total = len(dates_list)
    
    for idx, date_str in enumerate(dates_list):
        try:
            d = ee.Date(date_str)
            gpm = ee.ImageCollection('JAXA/GPM_L3/GSMaP/v8/operational').select('hourlyPrecipRateGC')
            
            # Create multiband image for the window range
            rain_bands = [
                gpm.filterDate(d.advance(-i, 'day'), d).sum().unmask(0).rename(f'Rn{i}')
                for i in range(min_d, max_d + 1)
            ]
            combined = ee.Image.cat(rain_bands)
            
            # Label polygons based on points
            todays_points = points_fc.filter(ee.Filter.eq(date_col, date_str))
            
            def map_polygons(poly):
                count = todays_points.filterBounds(poly.geometry()).size()
                return poly.set({'P/A': ee.Algorithms.If(count.gt(0), 1, 0), 'date': date_str})
            
            labeled_polys = poly_fc.map(map_polygons)
            
            # Add numeric ID for mapping
            def add_numeric_id(feature):
                str_id = ee.String(feature.get('id'))
                num_str = str_id.replace(r'[^0-9]', '', 'g')
                num_val = ee.Algorithms.If(num_str.length().gt(0), ee.Number.parse(num_str), 0)
                return feature.set('NUM_ID', num_val)
                
            labeled_polys = labeled_polys.map(add_numeric_id)

            # Extract stats
            stats = combined.reduceRegions(
                collection=labeled_polys,
                reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
                scale=1000, tileScale=16
            )
            
            df_day = geemap.ee_to_df(stats)
            if not df_day.empty:
                # Rename columns
                rename_dict = {f'Rn{i}_{suffix}': f'Rn{i}_{m}'
                               for i in range(min_d, max_d + 1)
                               for suffix, m in [('mean', 'm'), ('stdDev', 's')]}
                df_day = df_day.rename(columns=rename_dict)
                results.append(df_day)
                
        except Exception as e:
            # Skip errors for single dates
            pass
            
        my_bar.progress((idx + 1) / total, text=f"Processing {date_str}...")
        
    my_bar.empty()
    
    if not results:
        return pd.DataFrame()
    
    final_df = pd.concat(results, ignore_index=True)
    
    # Deterministic sorting
    if 'date' in final_df.columns and 'id' in final_df.columns:
        final_df = final_df.sort_values(by=['date', 'id']).reset_index(drop=True)
        
    # Ensure static predictors exist
    for col in STATIC_PREDICTORS:
        if col not in final_df.columns: final_df[col] = 0
            
    return final_df.fillna(0)

def visualize_map(df, value_col, layer_name, asset_path, palette):
    """
    Helper to visualize results on the map
    """
    # 1. Clean IDs locally
    df_map = df.copy()
    df_map['NUM_ID_PY'] = df_map['id'].apply(lambda x: int(''.join(filter(str.isdigit, str(x))) or 0))
    
    # 2. Group by ID (take max value)
    df_flat = df_map.groupby('NUM_ID_PY')[value_col].max().reset_index()
    
    ids = df_flat['NUM_ID_PY'].tolist()
    vals = df_flat[value_col].tolist()
    
    # 3. Create GEE Image
    fc = ee.FeatureCollection(asset_path)
    
    def add_id(f):
        str_id = ee.String(f.get('id'))
        num = ee.Number.parse(str_id.replace(r'[^0-9]', '', 'g'))
        return f.set('NUM_ID', num)
    
    fc_mapped = fc.map(add_id)
    fc_img = fc_mapped.reduceToImage(['NUM_ID'], ee.Reducer.first())
    
    result_img = fc_img.remap(ids, vals).rename('val')
    result_img = result_img.updateMask(result_img.gte(0))
    
    # 4. Create Map
    m = geemap.Map()
    m.centerObject(fc, 10)
    
    vis_params = {'min': 0, 'max': 1, 'palette': palette}
    if "Confusion" in layer_name:
         vis_params = {'min': 0, 'max': 3, 'palette': ['#D10E00', '#DF564D', '#DBEADD', '#006B0B']} # FP, TN, FN, TP
         
    m.addLayer(result_img, vis_params, layer_name)
    return m

# --- 5. TABS LOGIC ---

tab1, tab2, tab3 = st.tabs(["🚀 Calibration", "⚖️ Validation", "🔮 Prediction"])

# --- TAB 1: CALIBRATION ---
with tab1:
    st.subheader("Model Training & Optimization")
    
    if st.button("Start Calibration", type="primary"):
        with st.spinner("Downloading data and finding best rainfall window..."):
            # 1. Download
            df = download_training_data(MIN_DAYS, MAX_DAYS, DATE_COLUMN, polygons_asset, points_asset)
            
            if df.empty:
                st.error("No data found or download failed.")
            else:
                st.session_state.training_df = df
                y = df['P/A']
                
                # 2. Optimize
                best_auc = 0
                best_w = MIN_DAYS
                
                for days in range(MIN_DAYS, MAX_DAYS + 1):
                    cols = [f'Rn{days}_m', f'Rn{days}_s']
                    if all(c in df.columns for c in cols):
                        X_tmp = df[cols].fillna(0)
                        # Fast training for optimization
                        rf = RandomForestClassifier(n_estimators=30, max_depth=7, class_weight='balanced', random_state=42)
                        rf.fit(X_tmp, y)
                        probs = rf.predict_proba(X_tmp)[:, 1]
                        score = auc(*roc_curve(y, probs)[:2])
                        
                        if score > best_auc:
                            best_auc = score
                            best_w = days
                
                st.session_state.best_days = best_w
                st.session_state.final_predictors = STATIC_PREDICTORS + [f'Rn{best_w}_m', f'Rn{best_w}_s']
                
                st.success(f"Optimization Done! Best Window: {best_w} Days (AUC: {best_auc:.3f})")
                
                # 3. Final Model
                X = df[st.session_state.final_predictors].fillna(0)
                rf_final = RandomForestClassifier(n_estimators=100, max_depth=10, oob_score=True, class_weight='balanced', random_state=42)
                rf_final.fit(X, y)
                st.session_state.model = rf_final
                
                # 4. Results
                probs = rf_final.oob_decision_function_[:, 1]
                df['calib_prob'] = probs
                
                col1, col2 = st.columns(2)
                with col1:
                    fpr, tpr, _ = roc_curve(y, probs)
                    fig = go.Figure(data=go.Scatter(x=fpr, y=tpr, fill='tozeroy', name='ROC'))
                    fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                    fig.update_layout(title="ROC Curve (OOB)", height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                with col2:
                    imp = pd.Series(rf_final.feature_importances_, index=st.session_state.final_predictors).sort_values()
                    fig2 = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation='h'))
                    fig2.update_layout(title="Feature Importance", height=300)
                    st.plotly_chart(fig2, use_container_width=True)

                st.markdown("### Calibration Map")
                m_calib = visualize_map(df, 'calib_prob', 'Calibration Probability', polygons_asset, ['white', 'green', 'yellow', 'red'])
                m_calib.to_streamlit(height=500)
                
                # CSV Download
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Calibration CSV", csv, "calibration.csv", "text/csv")

# --- TAB 2: VALIDATION ---
with tab2:
    st.subheader("Cross-Validation (10-Fold)")
    
    if st.button("Run Validation"):
        if st.session_state.model is None:
            st.warning("⚠️ Please run Calibration first!")
        else:
            with st.spinner("Running Cross-Validation..."):
                df = st.session_state.training_df
                X = df[st.session_state.final_predictors].fillna(0)
                y = df['P/A']
                
                y_probs = cross_val_predict(st.session_state.model, X, y, cv=StratifiedKFold(10), method='predict_proba')[:, 1]
                cv_auc = auc(*roc_curve(y, y_probs)[:2])
                
                st.metric("Validation AUC", f"{cv_auc:.4f}")
                
                df['valid_prob'] = y_probs
                m_valid = visualize_map(df, 'valid_prob', 'Validation Probability', polygons_asset, ['white', 'orange', 'red'])
                m_valid.to_streamlit(height=500)

# --- TAB 3: PREDICTION ---
with tab3:
    st.subheader(f"Forecast for: {FORECAST_DATE}")
    
    if st.button("Run Prediction"):
        if st.session_state.model is None:
            st.warning("⚠️ Please train the model in the Calibration tab first!")
        else:
            with st.spinner("Fetching live satellite data..."):
                target_d = ee.Date(FORECAST_DATE.strftime('%Y-%m-%d'))
                days = st.session_state.best_days
                
                # Check data availability
                gpm = ee.ImageCollection('JAXA/GPM_L3/GSMaP/v8/operational').select('hourlyPrecipRateGC')
                # Find closest previous image
                latest = gpm.filterDate('2000-01-01', target_d.advance(1, 'day')).sort('system:time_start', False).first()
                
                if latest:
                    found_ms = latest.get('system:time_start').getInfo()
                    found_date = ee.Date(found_ms)
                    st.info(f"Using rainfall accumulation ending on: {found_date.format('YYYY-MM-dd').getInfo()}")
                    
                    # Accumulate Rain
                    rain_img = gpm.filterDate(found_date.advance(-days, 'day'), found_date.advance(1, 'day')).sum().unmask(0)
                    
                    # Process Prediction Polygons
                    pred_fc = ee.FeatureCollection(prediction_asset)
                    
                    # Add numeric ID
                    def add_num_id(f):
                        s = ee.String(f.get('id'))
                        n = ee.Number.parse(s.replace(r'[^0-9]', '', 'g'))
                        return f.set('NUM_ID', n)
                    pred_fc = pred_fc.map(add_num_id)
                    
                    stats = rain_img.reduceRegions(
                        collection=pred_fc,
                        reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
                        scale=1000
                    )
                    
                    df_pred = geemap.ee_to_df(stats)
                    
                    # Map columns to model features
                    # The reduceRegions will create generic names like 'mean' or 'hourlyPrecipRateGC_mean'
                    # We need to map them to Rn{days}_m and Rn{days}_s
                    
                    col_mean = f'Rn{days}_m'
                    col_std = f'Rn{days}_s'
                    
                    # Attempt to find the columns
                    mean_candidates = [c for c in df_pred.columns if 'mean' in c]
                    std_candidates = [c for c in df_pred.columns if 'stdDev' in c]
                    
                    df_pred[col_mean] = df_pred[mean_candidates[0]] if mean_candidates else 0
                    df_pred[col_std] = df_pred[std_candidates[0]] if std_candidates else 0
                    
                    # Add static predictors (default to 0 if missing in shapefile, usually they should be there)
                    for col in STATIC_PREDICTORS:
                        if col not in df_pred.columns: df_pred[col] = 0
                        
                    # Predict
                    X_pred = df_pred[st.session_state.final_predictors].fillna(0)
                    probs = st.session_state.model.predict_proba(X_pred)[:, 1]
                    df_pred['SI'] = probs
                    
                    st.success(f"Prediction Complete. Max Risk Score: {probs.max():.2f}")
                    
                    m_pred = visualize_map(df_pred, 'SI', 'Landslide Susceptibility', prediction_asset, ['green', 'yellow', 'red'])
                    m_pred.to_streamlit(height=600)
                    
                    csv_pred = df_pred.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Prediction CSV", csv_pred, "prediction.csv", "text/csv")
                else:
                    st.error("No satellite data found for this date.")