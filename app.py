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

# --- 2. AUTHENTICATION (ROBUST) ---
import subprocess # Aggiungi questo import in alto se manca

# --- 2. AUTHENTICATION (CON POPUP DI SISTEMA) ---
def check_gee_auth():
    DEFAULT_PROJECT = 'stgee-dataset' 
    
    # 1. Controlla se siamo già autenticati nella sessione
    if 'is_authenticated' not in st.session_state:
        st.session_state.is_authenticated = False

    if st.session_state.is_authenticated:
        return True

    # 2. Prova i Secrets (per quando sarai sul Cloud)
    if "EARTHENGINE_TOKEN" in st.secrets:
        try:
            token_data = st.secrets["EARTHENGINE_TOKEN"]
            if isinstance(token_data, str):
                token_dict = json.loads(token_data)
                creds = Credentials.from_authorized_user_info(token_dict)
            else:
                creds = Credentials.from_authorized_user_info(dict(token_data))
            ee.Initialize(credentials=creds, project=DEFAULT_PROJECT)
            st.session_state.is_authenticated = True
            return True
        except Exception:
            pass

    # 3. Prova l'inizializzazione Locale standard
    try:
        ee.Initialize(project=DEFAULT_PROJECT)
        st.session_state.is_authenticated = True
        return True
    except Exception:
        pass

    # 4. INTERFACCIA DI LOGIN (Se tutto il resto fallisce)
    st.warning("⚠️ Accesso a Google Earth Engine non rilevato.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Bottone per lanciare il processo
        if st.button("🔐 Autentica ora", type="primary"):
            try:
                # Questo comando lancia l'autenticazione nel sistema operativo
                # Dovrebbe aprire il browser predefinito del tuo PC
                process = subprocess.Popen(["earthengine", "authenticate"], shell=True)
                st.info("🌐 Si dovrebbe essere aperta una finestra del browser. Completa il login lì.")
            except Exception as e:
                st.error(f"Impossibile aprire il browser automaticamente: {e}")

    with col2:
        st.markdown("""
        **Istruzioni:**
        1. Clicca il pulsante a sinistra.
        2. Se si apre una pagina web, accetta i termini.
        3. Torna qui e clicca il pulsante **"Ricarica App"** qui sotto.
        
        *Se il pulsante non apre nulla:*
        Apri il terminale nero dove hai lanciato streamlit e scrivi: `earthengine authenticate`
        """)

    # Bottone per ricaricare la pagina dopo aver fatto il login
    if st.button("🔄 Ho completato il login! Ricarica App"):
        st.rerun()
    
    # Ferma l'app finché non sei loggato
    st.stop()

# Esegui il controllo
check_gee_auth()

# --- 3. SESSION STATE INITIALIZATION ---
if 'analysis_started' not in st.session_state:
    st.session_state.analysis_started = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'best_days' not in st.session_state:
    st.session_state.best_days = 1
if 'final_predictors' not in st.session_state:
    st.session_state.final_predictors = []
if 'training_df' not in st.session_state:
    st.session_state.training_df = None

# --- 4. SIDEBAR CONFIGURATION ---
st.sidebar.header("1. Data Configuration")
polygons_asset = st.sidebar.text_input("Polygons Asset", "projects/stgee-dataset/assets/export_predictors_polygons2")
points_asset = st.sidebar.text_input("Points Asset", "projects/stgee-dataset/assets/pointsDate")
prediction_asset = st.sidebar.text_input("Prediction Asset", "projects/stgee-dataset/assets/export_predictors_polygons2")

DATE_COLUMN = st.sidebar.text_input("Date Column Name", "formatted_date")
LANDSLIDE_COLUMN = st.sidebar.text_input("Landslide ID Column", "id")

st.sidebar.header("2. Model Parameters")
MIN_DAYS = st.sidebar.number_input("Min Rain Days", 1, 60, 1)
MAX_DAYS = st.sidebar.number_input("Max Rain Days", 1, 60, 30)
FORECAST_DATE = st.sidebar.date_input("Forecast Date", pd.to_datetime("2025-11-26"))
STATIC_PREDICTORS = ['Relief_mea', 'S_mean', 'VCv_mean', 'Hill_mean', 'NDVI_mean']

# --- 5. CORE FUNCTIONS ---
@st.cache_data(ttl=3600, show_spinner=False)
def download_training_data(min_d, max_d, date_col, _polygons_path, _points_path):
    # (Codice identico alla versione precedente per il download dei dati)
    poly_fc = ee.FeatureCollection(_polygons_path)
    points_fc = ee.FeatureCollection(_points_path)
    raw_dates = points_fc.aggregate_array(date_col).distinct().getInfo()
    dates_list = [str(d)[:10] for d in raw_dates]
    results = []
    
    progress_text = "Downloading Data..."
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
    if 'date' in final_df.columns and 'id' in final_df.columns: final_df = final_df.sort_values(by=['date', 'id']).reset_index(drop=True)
    for col in STATIC_PREDICTORS:
        if col not in final_df.columns: final_df[col] = 0
    return final_df.fillna(0)

def visualize_map(df, value_col, layer_name, asset_path, palette):
    # Helper per visualizzare i risultati (uguale a prima)
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

# --- 6. MAIN APP LAYOUT & LOGIC ---

st.title("PySTGEE: Landslide Hazard Modeling")

# *** STATE 0: INITIAL LANDING PAGE ***
if not st.session_state.analysis_started:
    st.markdown("""
    **Welcome.** This tool allows you to calibrate a Random Forest model using rainfall data from Google Earth Engine.
    
    1. Configure your assets in the Sidebar.
    2. Click **Run Analysis** below to load the study area and enable the tools.
    """)
    
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        st.session_state.analysis_started = True
        st.rerun()

# *** STATE 1: DASHBOARD ACTIVE ***
else:
    # 1. SHOW BASE MAP IMMEDIATELY (This fixes "Non vedo la mappa")
    st.markdown("### Study Area Overview")
    try:
        # Load simple base map of polygons
        m_base = geemap.Map()
        study_area = ee.FeatureCollection(polygons_asset)
        m_base.centerObject(study_area, 10)
        # Add polygons in blue outline
        m_base.addLayer(study_area.style(**{'color': '0000FF', 'fillColor': '00000000'}), {}, "Study Area Polygons")
        m_base.to_streamlit(height=500)
    except Exception as e:
        st.error(f"Error loading base map: {e}")

    # 2. SHOW TABS
    st.divider()
    tab1, tab2, tab3 = st.tabs(["📊 Calibration", "⚖️ Validation", "🔮 Prediction"])

    # --- TAB 1: CALIBRATION ---
    with tab1:
        st.subheader("Model Training")
        st.info("Click 'Start Calibration' to download rainfall data and train the model.")
        
        if st.button("Start Calibration", type="primary"):
            with st.spinner("Processing..."):
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
                            score = auc(*roc_curve(y, rf.predict_proba(X_tmp)[:, 1])[:2])
                            if score > best_auc:
                                best_auc = score
                                best_w = days
                    
                    st.session_state.best_days = best_w
                    st.session_state.final_predictors = STATIC_PREDICTORS + [f'Rn{best_w}_m', f'Rn{best_w}_s']
                    
                    # Final Model
                    X = df[st.session_state.final_predictors].fillna(0)
                    rf_final = RandomForestClassifier(n_estimators=100, max_depth=10, oob_score=True, class_weight='balanced', random_state=42)
                    rf_final.fit(X, y)
                    st.session_state.model = rf_final
                    
                    # Plots
                    probs = rf_final.oob_decision_function_[:, 1]
                    col1, col2 = st.columns(2)
                    with col1:
                        fpr, tpr, _ = roc_curve(y, probs)
                        fig = go.Figure(go.Scatter(x=fpr, y=tpr, fill='tozeroy', name='ROC'))
                        fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                        fig.update_layout(title=f"ROC Curve (AUC: {best_auc:.3f})", height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        imp = pd.Series(rf_final.feature_importances_, index=st.session_state.final_predictors).sort_values()
                        fig2 = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation='h'))
                        fig2.update_layout(title="Feature Importance", height=300)
                        st.plotly_chart(fig2, use_container_width=True)

                    st.success(f"Calibration Done! Best Window: {best_w} Days")
                    
                    # Result Map
                    st.markdown("#### Calibration Results Map")
                    df['calib_prob'] = probs
                    m_calib = visualize_map(df, 'calib_prob', 'Calib Prob', polygons_asset, ['white', 'green', 'yellow', 'red'])
                    m_calib.to_streamlit(height=500)

    # --- TAB 2: VALIDATION ---
    with tab2:
        if st.button("Run Validation"):
            if st.session_state.model is None:
                st.warning("Train the model first!")
            else:
                with st.spinner("Validating..."):
                    df = st.session_state.training_df
                    X = df[st.session_state.final_predictors].fillna(0)
                    y = df['P/A']
                    probs = cross_val_predict(st.session_state.model, X, y, cv=StratifiedKFold(10), method='predict_proba')[:, 1]
                    
                    st.metric("CV AUC", f"{auc(*roc_curve(y, probs)[:2]):.4f}")
                    df['valid_prob'] = probs
                    m_valid = visualize_map(df, 'valid_prob', 'Validation', polygons_asset, ['white', 'orange', 'red'])
                    m_valid.to_streamlit(height=500)

    # --- TAB 3: PREDICTION ---
    with tab3:
        st.write(f"Forecast Date: **{FORECAST_DATE}**")
        if st.button("Run Prediction"):
            if st.session_state.model is None:
                st.warning("Train the model first!")
            else:
                with st.spinner("Predicting..."):
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
                        
                        m_cand = [c for c in df_pred.columns if 'mean' in c]
                        s_cand = [c for c in df_pred.columns if 'stdDev' in c]
                        
                        df_pred[col_mean] = df_pred[m_cand[0]] if m_cand else 0
                        df_pred[col_std] = df_pred[s_cand[0]] if s_cand else 0
                        
                        for col in STATIC_PREDICTORS:
                            if col not in df_pred.columns: df_pred[col] = 0
                            
                        X_pred = df_pred[st.session_state.final_predictors].fillna(0)
                        probs = st.session_state.model.predict_proba(X_pred)[:, 1]
                        df_pred['SI'] = probs
                        
                        st.success(f"Max Risk: {probs.max():.2f}")
                        m_pred = visualize_map(df_pred, 'SI', 'Prediction', prediction_asset, ['green', 'yellow', 'red'])
                        m_pred.to_streamlit(height=600)
                        
                        csv = df_pred.to_csv(index=False).encode('utf-8')
                        st.download_button("Download CSV", csv, "prediction.csv", "text/csv")
                    else:
                        st.error("No satellite data found.")
    
    # Bottone per ricominciare (facoltativo)
    if st.button("Reset Analysis"):
        st.session_state.analysis_started = False
        st.rerun()
