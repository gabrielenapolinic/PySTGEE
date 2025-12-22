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
import re

# --- CONFIGURAZIONE UTENTE ---
EE_PROJECT = 'stgee-dataset'
POLYGONS_ASSET = "projects/stgee-dataset/assets/export_predictors_polygons2"
POINTS_ASSET = "projects/stgee-dataset/assets/pointsDate"
PREDICTION_ASSET = "projects/stgee-dataset/assets/export_predictors_polygons2"

DATE_COLUMN = 'formatted_date'
LANDSLIDE_COLUMN = 'id'
CSV_EXPORT_MODE = 'BEST_ONLY'
FORECAST_DATE_FIXED = '2025-11-26'

STATIC_PREDICTORS = ['Relief_mea', 'S_mean', 'VCv_mean', 'Hill_mean', 'NDVI_mean']
MIN_DAYS = 1
MAX_DAYS = 30

# --- PALETTE ---
VIS_PALETTE = [
    '#006b0b', '#1b7b25', '#4e9956', '#dbeadd', '#ffffff',
    '#f0b2ae', '#eb958f', '#df564d', '#d10e00'
]
PALETTE_CONFUSION = ['#D10E00', '#DF564D', '#DBEADD', '#006B0B']

# --- GESTIONE LOG ---
def log(message):
    if 'logs' not in st.session_state:
        st.session_state['logs'] = []
    st.session_state['logs'].append(str(message))

def render_log_console():
    st.markdown("### OPERATION LOG")
    if 'logs' in st.session_state and st.session_state['logs']:
        st.code("\n".join(st.session_state['logs']), language="text")
    else:
        st.code("Waiting for analysis...", language="text")

# --- AUTENTICAZIONE ---
@st.cache_resource
def initialize_ee():
    try:
        if "gcp_service_account" in st.secrets:
            creds_dict = dict(st.secrets["gcp_service_account"])
            credentials = service_account.Credentials.from_service_account_info(
                creds_dict, scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            ee.Initialize(credentials, project=EE_PROJECT)
            return True
        else:
            ee.Initialize(project=EE_PROJECT)
            return True
    except Exception as e:
        st.error(f"Authentication failed: {e}")
        return False

# --- FUNZIONI UTILI ---
def add_numeric_id(feature):
    # Usa un fallback '0' se l'ID manca per evitare crash
    str_id = ee.String(feature.get('id', '0'))
    num_str = str_id.replace(r'[^0-9]', '', 'g')
    num_val = ee.Algorithms.If(num_str.length().gt(0), ee.Number.parse(num_str), 0)
    return feature.set('NUM_ID', num_val)

def calc_confusion_class(row, pred_col, true_col='P/A'):
    p = int(row[pred_col])
    t = int(row[true_col])
    if p == 1 and t == 0: return 0 # FP
    if p == 0 and t == 0: return 1 # TN
    if p == 0 and t == 1: return 2 # FN
    if p == 1 and t == 1: return 3 # TP
    return 1

def calculate_advanced_metrics(y_true, y_probs):
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
        'youden': youden_scores[best_idx],
        'fpr': fpr, 'tpr': tpr, 'best_idx': best_idx, 
        'y_pred_opt': y_pred_opt
    }

def filter_dataframe_for_export(df):
    if CSV_EXPORT_MODE == 'ALL_DATA': return df
    cols_to_keep = ['id', 'date', 'P/A', 'NUM_ID']
    if 'final_predictors' in st.session_state:
        cols_to_keep.extend(st.session_state['final_predictors'])
    result_cols = ['calib_prob', 'calib_pred', 'conf_class', 'valid_prob', 'valid_pred', 'valid_conf', 'SI', 'Prediction']
    for rc in result_cols:
        if rc in df.columns: cols_to_keep.append(rc)
    final_cols = [c for c in cols_to_keep if c in df.columns]
    return df[final_cols]

# --- MOTORE DI TRAINING ---
@st.cache_data(ttl=3600, show_spinner=False)
def download_training_data():
    landPoints = ee.FeatureCollection(POINTS_ASSET)
    predictors_polygons = ee.FeatureCollection(POLYGONS_ASSET).map(add_numeric_id)
    raw_dates = landPoints.aggregate_array(DATE_COLUMN).distinct().getInfo()
    dates_list = [str(d)[:10] for d in raw_dates]
    
    meta_info = f"Event Dates found: {len(dates_list)}\nRetrieving rainfall data for windows {MIN_DAYS}-{MAX_DAYS} days..."
    
    def process_date(date_str):
        try:
            d = ee.Date(date_str)
            gpm = ee.ImageCollection('JAXA/GPM_L3/GSMaP/v8/operational').select('hourlyPrecipRateGC')
            rain_bands = [
                gpm.filterDate(d.advance(-i, 'day'), d).sum().unmask(0).rename(f'Rn{i}')
                for i in range(MIN_DAYS, MAX_DAYS + 1)
            ]
            combined = ee.Image.cat(rain_bands)
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
            rename_dict = {f'Rn{i}_{suffix}': f'Rn{i}_{m}'
                           for i in range(MIN_DAYS, MAX_DAYS + 1)
                           for suffix, m in [('mean', 'm'), ('stdDev', 's')]}
            return df_day.rename(columns=rename_dict)
        except Exception: return None

    results = []
    prog_bar = st.progress(0)
    status = st.empty()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_date, d): d for d in dates_list}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res is not None: results.append(res)
            completed += 1
            prog_bar.progress(completed / len(dates_list))
            status.text(f"...processed {completed}/{len(dates_list)} dates")
            
    prog_bar.empty()
    status.empty()

    if not results: return pd.DataFrame(), meta_info
    final_df = pd.concat(results, ignore_index=True)
    if 'date' in final_df.columns and 'id' in final_df.columns:
        final_df = final_df.sort_values(by=['date', 'id']).reset_index(drop=True)
    for col in STATIC_PREDICTORS:
        if col not in final_df.columns: final_df[col] = 0
        
    return final_df.fillna(0), meta_info

# --- MOTORE DI PREDIZIONE ---
def get_prediction_data_dynamic(target_date, best_days_n):
    target_date_ee = ee.Date(target_date)
    prediction_area_shp = ee.FeatureCollection(PREDICTION_ASSET).map(add_numeric_id)
    gpm = ee.ImageCollection('JAXA/GPM_L3/GSMaP/v8/operational').select('hourlyPrecipRateGC')

    available_col = gpm.filterDate('2000-01-01', target_date_ee.advance(1, 'day')) \
                       .sort('system:time_start', False).limit(1)
    
    has_data = available_col.size().getInfo() > 0
    log_msgs = ["-"*45, "SATELLITE DATA REPORT", "-"*45, f"Requested Date : {target_date}"]
    
    if has_data:
        latest_img = available_col.first()
        found_date_ms = latest_img.get('system:time_start').getInfo()
        found_date = ee.Date(found_date_ms)
        found_date_str = found_date.format('YYYY-MM-dd').getInfo()
        
        log_msgs.append(f"Source Image   : JAXA GSMaP v8 Operational")
        log_msgs.append(f"Available Date : {found_date_str}")
        
        if found_date_str != target_date:
            diff = ee.Date(target_date).difference(found_date, 'day').getInfo()
            log_msgs.append(f"STATUS         : FALLBACK ACTIVATED")
            log_msgs.append(f"   (Data lag of {int(diff)} days)")
        else:
            log_msgs.append(f"STATUS         : EXACT MATCH")
        
        log_msgs.append("-" * 45)
        rain_img = gpm.filterDate(found_date.advance(-best_days_n, 'day'), found_date.advance(1, 'day')) \
                      .sum().unmask(0).rename(f'Rn{best_days_n}')
    else:
        log_msgs.append("STATUS         : NO DATA FOUND (Using 0 Rain)")
        log_msgs.append("-" * 45)
        rain_img = ee.Image.constant(0).rename(f'Rn{best_days_n}')

    stats = rain_img.reduceRegions(
        collection=prediction_area_shp,
        reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
        scale=1000, tileScale=16
    )
    df = geemap.ee_to_df(stats)
    
    target_mean = f'Rn{best_days_n}_m'
    target_std = f'Rn{best_days_n}_s'
    
    for col in [f'Rn{best_days_n}_mean', 'hourlyPrecipRateGC_mean', 'mean']:
        if col in df.columns: df[target_mean] = df[col]; break
    if target_mean not in df.columns: df[target_mean] = 0
    for col in [f'Rn{best_days_n}_stdDev', 'hourlyPrecipRateGC_stdDev', 'stdDev']:
        if col in df.columns: df[target_std] = df[col]; break
    if target_std not in df.columns: df[target_std] = 0

    for col in STATIC_PREDICTORS:
        if col not in df.columns: df[col] = 0
        
    return df, log_msgs

# --- GESTIONE LAYER MAPPA (CRASH FIX) ---
def get_map_layers(df, val_col, layer_name, palette):
    """
    Prepara il layer ma NON lo aggiunge direttamente alla mappa.
    FIX: Converte esplicitamente i tipi Numpy in tipi Python nativi per evitare crash di GEE JSON.
    """
    def clean_id_py(val):
        s = str(val)
        digits = re.sub(r'[^0-9]', '', s)
        # Assicura che ritorni un int Python puro
        return int(digits) if digits else 0

    df_map = df.copy()
    df_map['NUM_ID_PY'] = df_map['id'].apply(clean_id_py)

    if layer_name.startswith("Confusion"):
        df_flat = df_map.sort_values('date').drop_duplicates(subset='NUM_ID_PY', keep='last')
    else:
        df_flat = df_map.groupby('NUM_ID_PY')[val_col].max().reset_index()

    # FIX CRITICO: Forziamo la conversione in liste Python pure (int e float)
    # GEE crasha se riceve numpy.int64 o numpy.float64
    id_list = [int(x) for x in df_flat['NUM_ID_PY'].values]
    val_list = [float(x) for x in df_flat[val_col].values]

    polygons_base = ee.FeatureCollection(PREDICTION_ASSET).map(add_numeric_id)
    polygons_img = polygons_base.reduceToImage(properties=['NUM_ID'], reducer=ee.Reducer.first())
    result_img = polygons_img.remap(id_list, val_list).rename('value')
    result_img = result_img.updateMask(result_img.gte(0))

    vis = {'palette': palette, 'min': 0, 'max': 3 if layer_name.startswith("Confusion") else 1}
    return result_img, vis

# --- MAIN APP ---
def run_app():
    st.title("PySTGEE: Landslide Modeling")

    # Inizializza Session State
    if 'analysis_active' not in st.session_state:
        st.session_state['analysis_active'] = False
    if 'logs' not in st.session_state:
        st.session_state['logs'] = []
    
    # --- FASE 1: LAUNCHER (Il bottone "Run Analysis") ---
    if not st.session_state['analysis_active']:
        if st.button("Run Analysis", type="primary"):
            st.session_state['analysis_active'] = True
            st.rerun() # Ricarica per mostrare la Fase 2

    # --- FASE 2: INTERFACCIA COMPLETA ---
    else:
        # 1. Autenticazione
        if not initialize_ee(): st.stop()

        # 2. Console Log
        render_log_console()

        # 3. Controlli
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            btn_calib = st.button("Run Calibration")
        with col2:
            btn_valid = st.button("Run Validation")
        with col3:
            btn_pred = st.button("Run Prediction")
        with col4:
            f_date = st.date_input("Date", value=pd.to_datetime('today'), label_visibility="collapsed")

        # 4. Inizializza variabili di stato
        if 'model' not in st.session_state: st.session_state['model'] = None
        if 'best_window' not in st.session_state: st.session_state['best_window'] = None
        if 'final_predictors' not in st.session_state: st.session_state['final_predictors'] = []
        if 'training_df' not in st.session_state: st.session_state['training_df'] = None
        if 'active_layers' not in st.session_state: st.session_state['active_layers'] = [] 
        if 'metrics_html' not in st.session_state: st.session_state['metrics_html'] = ""
        if 'charts' not in st.session_state: st.session_state['charts'] = None

        # 5. LOGICA DEI BOTTONI
        
        # --- CALIBRATION ---
        if btn_calib:
            st.session_state['logs'] = []
            log("Loading assets...")
            with st.spinner("Downloading Data..."):
                df, meta = download_training_data()
                st.session_state['training_df'] = df
                log(meta)
            
            if not df.empty:
                y = df['P/A']
                best_auc = 0
                best_days = MIN_DAYS
                log(f"Starting Optimization: Scanning windows {MIN_DAYS}-{MAX_DAYS} days...")
                log("-" * 30)
                
                prog = st.progress(0)
                for days in range(MIN_DAYS, MAX_DAYS + 1):
                    cols = [f'Rn{days}_m', f'Rn{days}_s']
                    if not all(c in df.columns for c in cols): continue
                    X_temp = df[cols].fillna(0)
                    rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, class_weight='balanced')
                    rf.fit(X_temp, y)
                    probs = rf.predict_proba(X_temp)[:, 1]
                    fpr, tpr, _ = roc_curve(y, probs)
                    score = auc(fpr, tpr)
                    log(f"   > Day {days}: AUC = {score:.4f}")
                    if score > best_auc: best_auc, best_days = score, days
                    prog.progress((days - MIN_DAYS) / (MAX_DAYS - MIN_DAYS))
                prog.empty()

                st.session_state['best_window'] = best_days
                st.session_state['final_predictors'] = STATIC_PREDICTORS + [f'Rn{best_days}_m', f'Rn{best_days}_s']
                log("-" * 30)
                log(f"FINAL: Best Window: {best_days} Days | Max AUC: {best_auc:.4f}")
                log("Training final model...")

                X = df[st.session_state['final_predictors']].fillna(0)
                rf_final = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, oob_score=True, class_weight='balanced')
                rf_final.fit(X, y)
                st.session_state['model'] = rf_final

                # Calcolo Metriche e Layer
                df['calib_prob'] = rf_final.predict_proba(X)[:, 1]
                m = calculate_advanced_metrics(y, df['calib_prob'])
                df['calib_pred'] = m['y_pred_opt']
                df['conf_class'] = df.apply(lambda r: calc_confusion_class(r, 'calib_pred'), axis=1)

                st.session_state['metrics_html'] = f"<b>Calibration</b>: AUC={m['auc']:.3f} | Acc={m['acc']:.3f} | F1={m['f1']:.3f}"
                
                # PREPARA I LAYER
                try:
                    l1, v1 = get_map_layers(df, 'calib_prob', 'Calibration Map', VIS_PALETTE)
                    l2, v2 = get_map_layers(df, 'conf_class', 'Confusion Calibration', PALETTE_CONFUSION)
                    st.session_state['active_layers'] = [
                        (l1, v1, 'Calibration Map'),
                        (l2, v2, 'Confusion Calibration')
                    ]
                except Exception as e:
                    log(f"ERROR creating map layers: {e}")
                
                # Grafici
                imp = pd.Series(rf_final.feature_importances_, index=st.session_state['final_predictors']).sort_values()
                fig = make_subplots(rows=2, cols=1, subplot_titles=("Feature Importance", "Confusion Matrix"))
                fig.add_trace(go.Bar(x=imp.values, y=imp.index, orientation='h'), row=1, col=1)
                cm = confusion_matrix(y, m['y_pred_opt'])
                fig.add_trace(go.Heatmap(z=cm, x=['Pred:0', 'Pred:1'], y=['True:0', 'True:1'], colorscale='Blues', texttemplate="%{z}"), row=2, col=1)
                fig.update_layout(height=500, showlegend=False)
                st.session_state['charts'] = fig
                st.rerun()

        # --- VALIDATION ---
        if btn_valid:
            if not st.session_state['model']: st.error("Train Model First!")
            else:
                log("Running Validation...")
                df = st.session_state['training_df']
                X = df[st.session_state['final_predictors']].fillna(0)
                y = df['P/A']
                y_probs = cross_val_predict(st.session_state['model'], X, y, cv=StratifiedKFold(n_splits=10), method='predict_proba')[:, 1]
                m = calculate_advanced_metrics(y, y_probs)
                log(f"Validation Done. AUC: {m['auc']:.4f}")
                
                df['valid_prob'] = y_probs
                df['valid_pred'] = m['y_pred_opt']
                df['valid_conf'] = df.apply(lambda r: calc_confusion_class(r, 'valid_pred'), axis=1)
                
                st.session_state['metrics_html'] = f"<b>Validation</b>: AUC={m['auc']:.3f} | F1={m['f1']:.3f}"
                try:
                    l1, v1 = get_map_layers(df, 'valid_prob', 'Validation Map', VIS_PALETTE)
                    l2, v2 = get_map_layers(df, 'valid_conf', 'Confusion Validation', PALETTE_CONFUSION)
                    st.session_state['active_layers'] = [
                        (l1, v1, 'Validation Map'),
                        (l2, v2, 'Confusion Validation')
                    ]
                except Exception as e:
                    log(f"ERROR creating map layers: {e}")
                st.rerun()

        # --- PREDICTION ---
        if btn_pred:
            if not st.session_state['model']: st.error("Train Model First!")
            else:
                with st.spinner("Processing..."):
                    df_pred, logs = get_prediction_data_dynamic(f_date.strftime('%Y-%m-%d'), st.session_state['best_window'])
                    for l in logs: log(l)
                    X_pred = df_pred[st.session_state['final_predictors']].fillna(0)
                    probs = st.session_state['model'].predict_proba(X_pred)[:, 1]
                    df_pred['SI'] = probs
                    log(f"Max Risk: {probs.max():.2f}")
                    
                    st.session_state['metrics_html'] = f"<b>Prediction</b>: Max Risk={probs.max():.2f}"
                    try:
                        l_pred, v_pred = get_map_layers(df_pred, 'SI', 'Prediction Map', VIS_PALETTE)
                        st.session_state['active_layers'] = [(l_pred, v_pred, 'Prediction Map')]
                    except Exception as e:
                        log(f"ERROR creating map layers: {e}")
                    st.rerun()

        # 6. RENDERIZZAZIONE FINALE
        
        # Risultati
        if st.session_state['metrics_html']:
            st.markdown(st.session_state['metrics_html'], unsafe_allow_html=True)
        if st.session_state['charts']:
            st.plotly_chart(st.session_state['charts'], use_container_width=True)

        # --- MAPPA ---
        try:
            m = geemap.Map(height=600)
            
            # Area di Studio
            predictors_polygons = ee.FeatureCollection(POLYGONS_ASSET)
            m.centerObject(predictors_polygons, 10)
            m.addLayer(predictors_polygons.style(**{'color': 'gray', 'fillColor': '00000000'}), {}, 'Study Area')
            
            # Layer Attivi
            for layer, vis, name in st.session_state['active_layers']:
                m.addLayer(layer, vis, name)
                if not name.startswith("Confusion"):
                    m.add_colorbar(vis, label=name)
            
            m.to_streamlit()
        except Exception as e:
            st.error(f"Error rendering map: {e}")
