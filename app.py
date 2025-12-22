import streamlit as st
import ee
import geemap.foliumap as geemap
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score, cohen_kappa_score, accuracy_score
import json

# --- CONFIGURATION & CONSTANTS ---
st.set_page_config(layout="wide", page_title="PySTGEE Web")

# Asset Paths (Fixed)
EE_PROJECT = 'stgee-dataset'
POLYGONS_ASSET = "projects/stgee-dataset/assets/export_predictors_polygons2"
POINTS_ASSET = "projects/stgee-dataset/assets/pointsDate"
PREDICTION_ASSET = "projects/stgee-dataset/assets/export_predictors_polygons2"
DATE_COLUMN = 'formatted_date'
STATIC_PREDICTORS = ['Relief_mea', 'S_mean', 'VCv_mean', 'Hill_mean', 'NDVI_mean']
VIS_PALETTE = ['#006b0b', '#1b7b25', '#4e9956', '#dbeadd', '#ffffff', '#f0b2ae', '#eb958f', '#df564d', '#d10e00']

# --- AUTHENTICATION (Service Account) ---
# Allows public users to access the app without logging in personally.
try:
    # Load secrets from Streamlit Cloud configuration
    service_account = st.secrets["EARTHENGINE_TOKEN"]
    credentials = ee.ServiceAccountCredentials(service_account["client_email"], key_data=json.dumps(dict(service_account)))
    ee.Initialize(credentials, project=EE_PROJECT)
except Exception as e:
    st.error(f"GEE Auth Failed: {e}")
    st.stop()

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Configuration")
min_days = st.sidebar.number_input("Min Search Window (Days)", 1, 30, 1)
max_days = st.sidebar.number_input("Max Search Window (Days)", 1, 30, 30)

# --- HELPER FUNCTIONS ---

@st.cache_data(ttl=3600)
def fetch_training_data(min_d, max_d):
    """
    Downloads rainfall data for defined window range. 
    Cached to prevent re-downloading on every interaction.
    """
    landPoints = ee.FeatureCollection(POINTS_ASSET)
    raw_polygons = ee.FeatureCollection(POLYGONS_ASSET)
    
    # Extract unique dates from points
    raw_dates = landPoints.aggregate_array(DATE_COLUMN).distinct().getInfo()
    dates_list = [str(d)[:10] for d in raw_dates]
    
    results = []
    
    # Progress bar setup
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_dates = len(dates_list)
    
    for idx, date_str in enumerate(dates_list):
        try:
            status_text.text(f"Processing date: {date_str} ({idx+1}/{total_dates})")
            d = ee.Date(date_str)
            gpm = ee.ImageCollection('JAXA/GPM_L3/GSMaP/v8/operational').select('hourlyPrecipRateGC')

            # Create bands for each day interval
            rain_bands = [
                gpm.filterDate(d.advance(-i, 'day'), d).sum().unmask(0).rename(f'Rn{i}')
                for i in range(min_d, max_d + 1)
            ]
            combined = ee.Image.cat(rain_bands)

            # Label polygons (Presence/Absence)
            todays_points = landPoints.filter(ee.Filter.eq(DATE_COLUMN, date_str))
            
            # Use raw GEE for labeling to speed up
            def set_pa(poly):
                count = todays_points.filterBounds(poly.geometry()).size()
                return poly.set({'P/A': ee.Algorithms.If(count.gt(0), 1, 0), 'date': date_str})

            labeled_polys = raw_polygons.map(set_pa)

            # Reduce regions (Extract Stats)
            stats = combined.reduceRegions(
                collection=labeled_polys,
                reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
                scale=1000, tileScale=16
            )
            
            df_day = geemap.ee_to_df(stats)
            
            # Rename columns to standardized format
            if not df_day.empty:
                rename_dict = {f'Rn{i}_{suffix}': f'Rn{i}_{m}' 
                               for i in range(min_d, max_d + 1) 
                               for suffix, m in [('mean', 'm'), ('stdDev', 's')]}
                df_day = df_day.rename(columns=rename_dict)
                results.append(df_day)
                
        except Exception:
            continue
        
        progress_bar.progress((idx + 1) / total_dates)

    status_text.empty()
    progress_bar.empty()

    if not results: return pd.DataFrame()
    
    final_df = pd.concat(results, ignore_index=True)
    # Deterministic sorting
    if 'date' in final_df.columns and 'id' in final_df.columns:
        final_df = final_df.sort_values(by=['date', 'id']).reset_index(drop=True)
        
    return final_df.fillna(0)

def train_and_optimize(df, min_d, max_d):
    """Iterates through day windows to find best AUC, then trains final RF."""
    y = df['P/A']
    best_auc = 0
    best_days = min_d
    
    # Optimization Loop
    for days in range(min_d, max_d + 1):
        cols = [f'Rn{days}_m', f'Rn{days}_s']
        if not all(c in df.columns for c in cols): continue
        
        X_temp = df[cols].fillna(0)
        rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, class_weight='balanced')
        rf.fit(X_temp, y)
        probs = rf.predict_proba(X_temp)[:, 1]
        fpr, tpr, _ = roc_curve(y, probs)
        curr_auc = auc(fpr, tpr)
        
        if curr_auc > best_auc:
            best_auc = curr_auc
            best_days = days

    # Final Training
    final_preds = STATIC_PREDICTORS + [f'Rn{best_days}_m', f'Rn{best_days}_s']
    # Ensure static cols exist
    for c in STATIC_PREDICTORS: 
        if c not in df.columns: df[c] = 0
            
    X = df[final_preds].fillna(0)
    rf_final = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, oob_score=True, class_weight='balanced')
    rf_final.fit(X, y)
    
    # Metrics
    y_probs = rf_final.oob_decision_function_[:, 1]
    fpr, tpr, thresholds = roc_curve(y, y_probs)
    best_idx = np.argmax(tpr - fpr)
    
    return {
        'model': rf_final,
        'best_days': best_days,
        'auc': auc(fpr, tpr),
        'predictors': final_preds,
        'probs': y_probs,
        'fpr': fpr, 'tpr': tpr, 'best_idx': best_idx
    }

# --- MAIN APP LAYOUT ---
st.title("🏔️ PySTGEE: Landslide Hazard Modeling")
st.markdown("Automated Spatio-Temporal GEE Modeling powered by Streamlit.")

# Session State Init
if 'df_train' not in st.session_state: st.session_state.df_train = None
if 'model_res' not in st.session_state: st.session_state.model_res = None

# 1. DATA LOADING
col1, col2 = st.columns([1, 3])
with col1:
    if st.button("1. Load & Process Data", use_container_width=True):
        with st.spinner("Retrieving GEE Data (this may take time)..."):
            st.session_state.df_train = fetch_training_data(min_days, max_days)
            st.success(f"Loaded {len(st.session_state.df_train)} rows.")

# 2. CALIBRATION
if st.session_state.df_train is not None:
    with col1:
        if st.button("2. Run Calibration", use_container_width=True):
            with st.spinner("Optimizing Rainfall Window..."):
                res = train_and_optimize(st.session_state.df_train, min_days, max_days)
                st.session_state.model_res = res
                
                # Add results to DF for map/download
                st.session_state.df_train['calib_prob'] = res['probs']

    # Show Results
    if st.session_state.model_res:
        res = st.session_state.model_res
        st.metric("Best Rainfall Window", f"{res['best_days']} Days")
        st.metric("Max AUC", f"{res['auc']:.4f}")
        
        # Plot ROC
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res['fpr'], y=res['tpr'], fill='tozeroy', name='ROC'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash='dash', color='black')))
        st.plotly_chart(fig, use_container_width=True)
        
        # Download CSV
        csv = st.session_state.df_train.to_csv(index=False).encode('utf-8')
        st.download_button("Download Calibration CSV", csv, "calibration.csv", "text/csv")

# 3. MAPPING
st.subheader("Interactive Map")
Map = geemap.Map(center=[40, 10], zoom=4) # Adjust center roughly to your area
Map.add_basemap("HYBRID")

# Add Study Area
try:
    study_area = ee.FeatureCollection(POLYGONS_ASSET)
    Map.addLayer(study_area.style(color='white', fillColor='00000000'), {}, "Study Area")
except:
    st.warning("Could not load Study Area polygons.")

# Add Results Layer if available
if st.session_state.df_train is not None and 'calib_prob' in st.session_state.df_train.columns:
    # Visualization Logic
    df = st.session_state.df_train
    
    # We need to link DF 'id' back to GEE Features for visualization
    # Note: In a full app, this needs careful ID matching. 
    # Here we simulate by re-uploading logic or using local geodataframe if small enough.
    # For GEE-Server side join, we normally upload the results as a table, but that requires write permissions.
    # PRO TIP: For Streamlit visualization of large datasets, use geemap.add_gdf if converting to GeoDataFrame
    # or color the vectors locally if small (<5000 polys).
    
    st.info("Map visualization requires linking Pandas results back to GEE geometries. (Simplified for this demo)")

Map.to_streamlit(height=600)
