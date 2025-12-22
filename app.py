import streamlit as st
import ee
import geemap.foliumap as geemap
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google.oauth2 import service_account
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score, cohen_kappa_score, accuracy_score
import concurrent.futures

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="PySTGEE Landslide Model", layout="wide", page_icon="🏔️")

# --- SESSION STATE INITIALIZATION ---
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'best_window' not in st.session_state:
    st.session_state['best_window'] = None
if 'final_predictors' not in st.session_state:
    st.session_state['final_predictors'] = []
if 'training_df' not in st.session_state:
    st.session_state['training_df'] = None

# --- USER CONFIGURATION (FROM NOTEBOOK CELL 1) ---
# You can make these inputs dynamic in the sidebar if you want
EE_PROJECT = 'stgee-dataset'
POLYGONS_ASSET = "projects/stgee-dataset/assets/export_predictors_polygons2"
POINTS_ASSET = "projects/stgee-dataset/assets/pointsDate"
PREDICTION_ASSET = "projects/stgee-dataset/assets/export_predictors_polygons2"
DATE_COLUMN = 'formatted_date'
LANDSLIDE_COLUMN = 'id'
STATIC_PREDICTORS = ['Relief_mea', 'S_mean', 'VCv_mean', 'Hill_mean', 'NDVI_mean']
MIN_DAYS = 1
MAX_DAYS = 30

# --- AUTHENTICATION ---
@st.cache_resource
def initialize_ee():
    """Authenticates and initializes Earth Engine."""
    try:
        # Try loading from Streamlit secrets
        if "gcp_service_account" in st.secrets:
            creds_dict = dict(st.secrets["gcp_service_account"])
            credentials = service_account.Credentials.from_service_account_info(
                creds_dict, 
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            ee.Initialize(credentials, project=EE_PROJECT)
            return True
        else:
            # Fallback for local testing if authenticated via CLI
            ee.Initialize(project=EE_PROJECT)
            return True
    except Exception as e:
        st.error(f"Authentication failed: {e}")
        return False

# --- HELPER FUNCTIONS ---
def add_numeric_id(feature):
    str_id = ee.String(feature.get('id'))
    num_str = str_id.replace(r'[^0-9]', '', 'g')
    num_val = ee.Algorithms.If(num_str.length().gt(0), ee.Number.parse(num_str), 0)
    return feature.set('NUM_ID', num_val)

def calculate_metrics(y_true, y_probs):
    fpr, tpr, roc_thresh = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    youden_scores = tpr - fpr
    best_idx = np.argmax(youden_scores)
    best_thresh = roc_thresh[best_idx]
    y_pred_opt = (y_probs >= best_thresh).astype(int)
    f1 = f1_score(y_true, y_pred_opt, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred_opt)
    acc = accuracy_score(y_true, y_pred_opt)
    return {
        'auc': roc_auc, 'best_thresh': best_thresh,
        'f1': f1, 'kappa': kappa, 'acc': acc,
        'fpr': fpr, 'tpr': tpr, 'best_idx': best_idx, 
        'y_pred_opt': y_pred_opt
    }

# --- DATA DOWNLOAD ENGINE (CACHED) ---
@st.cache_data(ttl=3600) # Cache data for 1 hour
def download_training_data():
    """Downloads training data using Earth Engine (Server-side logic)."""
    
    # Load Assets
    landPoints = ee.FeatureCollection(POINTS_ASSET)
    predictors_polygons = ee.FeatureCollection(POLYGONS_ASSET).map(add_numeric_id)
    
    # Get Dates
    raw_dates = landPoints.aggregate_array(DATE_COLUMN).distinct().getInfo()
    dates_list = [str(d)[:10] for d in raw_dates]
    
    st.write(f"📅 Found {len(dates_list)} event dates. Starting download...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    def process_date(date_str):
        try:
            d = ee.Date(date_str)
            gpm = ee.ImageCollection('JAXA/GPM_L3/GSMaP/v8/operational').select('hourlyPrecipRateGC')
            
            # Create rainfall bands for 1 to 30 days
            rain_bands = [
                gpm.filterDate(d.advance(-i, 'day'), d).sum().unmask(0).rename(f'Rn{i}')
                for i in range(MIN_DAYS, MAX_DAYS + 1)
            ]
            combined = ee.Image.cat(rain_bands)
            
            # Label Polygons (Presence/Absence)
            todays_points = landPoints.filter(ee.Filter.eq(DATE_COLUMN, date_str))
            
            def map_polygons(poly):
                count = todays_points.filterBounds(poly.geometry()).size()
                return poly.set({'P/A': ee.Algorithms.If(count.gt(0), 1, 0), 'date': date_str})
            
            labeled_polys = predictors_polygons.map(map_polygons)
            
            stats = combined.reduceRegions(
                collection=labeled_polys,
                reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
                scale=1000, tileScale=16
            )
            
            df_day = geemap.ee_to_df(stats)
            if df_day.empty: return None
            
            # Rename columns to match notebook logic (RnX_mean -> RnX_m)
            rename_dict = {f'Rn{i}_{suffix}': f'Rn{i}_{m}'
                           for i in range(MIN_DAYS, MAX_DAYS + 1)
                           for suffix, m in [('mean', 'm'), ('stdDev', 's')]}
            return df_day.rename(columns=rename_dict)
            
        except Exception as e:
            return None

    results = []
    # Use ThreadPool for concurrency
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_date, d): d for d in dates_list}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res is not None:
                results.append(res)
            completed += 1
            progress_bar.progress(completed / len(dates_list))
            status_text.text(f"Processed {completed}/{len(dates_list)} dates")

    if not results:
        return pd.DataFrame()

    final_df = pd.concat(results, ignore_index=True)
    
    # Deterministic Sorting
    if 'date' in final_df.columns and 'id' in final_df.columns:
        final_df = final_df.sort_values(by=['date', 'id']).reset_index(drop=True)
        
    # Ensure static predictors exist
    for col in STATIC_PREDICTORS:
        if col not in final_df.columns: final_df[col] = 0
        
    return final_df.fillna(0)

# --- PREDICTION DATA FETCHING ---
def fetch_prediction_data(target_date, best_days_n):
    """Fetches data for a specific date to make a map."""
    
    target_date_ee = ee.Date(target_date)
    prediction_area_shp = ee.FeatureCollection(PREDICTION_ASSET).map(add_numeric_id)
    gpm = ee.ImageCollection('JAXA/GPM_L3/GSMaP/v8/operational').select('hourlyPrecipRateGC')

    # Find most recent data logic
    available_col = gpm.filterDate('2000-01-01', target_date_ee.advance(1, 'day')) \
                       .sort('system:time_start', False).limit(1)
    
    if available_col.size().getInfo() > 0:
        latest_img = available_col.first()
        found_date = ee.Date(latest_img.get('system:time_start'))
        rain_img = gpm.filterDate(found_date.advance(-best_days_n, 'day'), found_date.advance(1, 'day')) \
                      .sum().unmask(0).rename(f'Rn{best_days_n}')
        st.info(f"Using rainfall data ending: {found_date.format('YYYY-MM-dd').getInfo()}")
    else:
        st.warning("No satellite data found. Using 0 rainfall.")
        rain_img = ee.Image.constant(0).rename(f'Rn{best_days_n}')

    stats = rain_img.reduceRegions(
        collection=prediction_area_shp,
        reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
        scale=1000, tileScale=16
    )
    
    df = geemap.ee_to_df(stats)
    
    # Renaming Logic
    target_mean = f'Rn{best_days_n}_m'
    target_std = f'Rn{best_days_n}_s'
    
    # Try different naming conventions from GEE
    for col in [f'Rn{best_days_n}_mean', 'hourlyPrecipRateGC_mean', 'mean']:
        if col in df.columns:
            df[target_mean] = df[col]
            break
    if target_mean not in df.columns: df[target_mean] = 0
            
    for col in [f'Rn{best_days_n}_stdDev', 'hourlyPrecipRateGC_stdDev', 'stdDev']:
        if col in df.columns:
            df[target_std] = df[col]
            break
    if target_std not in df.columns: df[target_std] = 0

    for col in STATIC_PREDICTORS:
        if col not in df.columns: df[col] = 0
        
    return df

# --- MAIN APP LAYOUT ---
def main():
    st.sidebar.title("PySTGEE Settings")
    
    if not initialize_ee():
        st.stop()
        
    # --- TABS FOR WORKFLOW ---
    tab1, tab2, tab3 = st.tabs(["1. Model Training", "2. Validation", "3. Forecast Map"])

    # --- TAB 1: TRAINING ---
    with tab1:
        st.header("Model Calibration")
        st.markdown("Downloads data and finds the optimal rainfall window (1-30 days).")
        
        if st.button("Start Training Process"):
            with st.spinner("Downloading and processing data (this may take a minute)..."):
                df = download_training_data()
                st.session_state['training_df'] = df
            
            if df is not None and not df.empty:
                st.success("Data Downloaded.")
                
                # Optimization Loop
                y = df['P/A']
                best_auc = 0
                best_days = MIN_DAYS
                
                progress_opt = st.progress(0)
                status_opt = st.empty()
                
                for days in range(MIN_DAYS, MAX_DAYS + 1):
                    cols = [f'Rn{days}_m', f'Rn{days}_s']
                    if not all(c in df.columns for c in cols): continue
                    
                    X_temp = df[cols].fillna(0)
                    rf = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42, class_weight='balanced')
                    rf.fit(X_temp, y)
                    probs = rf.predict_proba(X_temp)[:, 1]
                    fpr, tpr, _ = roc_curve(y, probs)
                    score = auc(fpr, tpr)
                    
                    if score > best_auc:
                        best_auc = score
                        best_days = days
                    
                    status_opt.text(f"Testing Day {days}... AUC: {score:.3f}")
                    progress_opt.progress((days - MIN_DAYS) / (MAX_DAYS - MIN_DAYS))
                
                st.session_state['best_window'] = best_days
                st.session_state['final_predictors'] = STATIC_PREDICTORS + [f'Rn{best_days}_m', f'Rn{best_days}_s']
                
                # Train Final Model
                st.subheader(f"🏆 Best Window: {best_days} Days (AUC: {best_auc:.4f})")
                
                X = df[st.session_state['final_predictors']].fillna(0)
                rf_final = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, oob_score=True, class_weight='balanced')
                rf_final.fit(X, y)
                
                st.session_state['model'] = rf_final
                
                # Plot Feature Importance
                imp = pd.Series(rf_final.feature_importances_, index=st.session_state['final_predictors']).sort_values()
                fig_imp = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation='h'))
                fig_imp.update_layout(title="Feature Importance", height=400)
                st.plotly_chart(fig_imp)
                
                st.success("Model Trained and Saved to Session!")

    # --- TAB 2: VALIDATION ---
    with tab2:
        st.header("Cross-Validation Results")
        if st.session_state['model'] is None:
            st.warning("Please train the model in Tab 1 first.")
        else:
            if st.button("Run Cross-Validation"):
                df = st.session_state['training_df']
                X = df[st.session_state['final_predictors']].fillna(0)
                y = df['P/A']
                
                with st.spinner("Running 10-Fold Cross Validation..."):
                    y_probs = cross_val_predict(st.session_state['model'], X, y, cv=StratifiedKFold(n_splits=10), method='predict_proba')[:, 1]
                    m = calculate_metrics(y, y_probs)
                
                # Metrics Display
                col1, col2, col3 = st.columns(3)
                col1.metric("AUC", f"{m['auc']:.4f}")
                col2.metric("Accuracy", f"{m['acc']:.4f}")
                col3.metric("F1 Score", f"{m['f1']:.4f}")
                
                # ROC Curve
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=m['fpr'], y=m['tpr'], fill='tozeroy', name='ROC'))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash='dash'), showlegend=False))
                fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                st.plotly_chart(fig_roc)
                
                # Confusion Matrix
                cm = confusion_matrix(y, m['y_pred_opt'])
                fig_cm = go.Figure(data=go.Heatmap(z=cm, x=['Pred 0', 'Pred 1'], y=['True 0', 'True 1'], colorscale='Blues', texttemplate="%{z}"))
                fig_cm.update_layout(title="Confusion Matrix")
                st.plotly_chart(fig_cm)

    # --- TAB 3: FORECAST ---
    with tab3:
        st.header("Generate Hazard Map")
        if st.session_state['model'] is None:
            st.warning("Please train the model in Tab 1 first.")
        else:
            forecast_date = st.date_input("Select Forecast Date", value=pd.to_datetime('today'))
            
            if st.button("Generate Map"):
                with st.spinner("Fetching data and predicting..."):
                    # Fetch Data
                    df_pred = fetch_prediction_data(forecast_date.strftime('%Y-%m-%d'), st.session_state['best_window'])
                    
                    # Predict
                    X_pred = df_pred[st.session_state['final_predictors']].fillna(0)
                    probs = st.session_state['model'].predict_proba(X_pred)[:, 1]
                    df_pred['probability'] = probs
                    
                    # Prepare for Map
                    # We need to map these values back to the geometry. 
                    # Streamlit handles maps best via geemap.
                    
                    # 1. Create an Earth Engine dictionary from the results {id: probability}
                    # Clean IDs to ensure they are integers
                    df_pred['id_clean'] = df_pred['id'].astype(str).str.replace(r'[^0-9]', '', regex=True).replace('', '0').astype(int)
                    
                    id_list = df_pred['id_clean'].tolist()
                    prob_list = df_pred['probability'].tolist()
                    
                    # 2. Remap the original polygon collection
                    polygons_base = ee.FeatureCollection(PREDICTION_ASSET).map(add_numeric_id)
                    
                    # Reduce to image for faster rendering
                    polys_img = polygons_base.reduceToImage(properties=['NUM_ID'], reducer=ee.Reducer.first())
                    
                    # Remap IDs to Probabilities
                    hazard_img = polys_img.remap(id_list, prob_list).rename('hazard')
                    
                    # Mask 0 values or background
                    hazard_img = hazard_img.updateMask(hazard_img.gte(0))
                    
                    # Visualization
                    vis_params = {
                        'min': 0, 
                        'max': 1, 
                        'palette': ['#006b0b', '#dbeadd', '#eb958f', '#d10e00']
                    }
                    
                    # Create Map
                    m = geemap.Map(center=[44.0, 10.0], zoom=10) # Adjust center roughly to Italy or your ROI
                    m.centerObject(polygons_base, 10)
                    m.addLayer(hazard_img, vis_params, "Landslide Hazard Probability")
                    
                    # Add colorbar
                    m.add_colorbar(vis_params, label="Probability")
                    
                    st.success(f"Map generated! Max Risk found: {max(prob_list):.2f}")
                    
                    # Render Map
                    m.to_streamlit(height=600)
                    
                    # Download Link
                    csv = df_pred[['id', 'probability']].to_csv(index=False).encode('utf-8')
                    st.download_button("Download Predictions (CSV)", csv, "predictions.csv", "text/csv")

if __name__ == "__main__":
    main()
