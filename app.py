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

# --- CONFIG ---
st.set_page_config(page_title="PySTGEE Dashboard", layout="wide")

# --- AUTHENTICATION ---
def check_auth():
    if 'is_auth' in st.session_state and st.session_state.is_auth:
        return True

    # 1. Try Secrets (Cloud)
    if "EARTHENGINE_TOKEN" in st.secrets:
        try:
            creds = Credentials.from_authorized_user_info(json.loads(st.secrets["EARTHENGINE_TOKEN"]))
            ee.Initialize(credentials=creds, project='stgee-dataset')
            st.session_state.is_auth = True
            return True
        except: pass

    # 2. Try Local Existing Token
    try:
        ee.Initialize(project='stgee-dataset')
        st.session_state.is_auth = True
        return True
    except: pass

    # 3. UI for Login
    st.warning("⚠️ GEE Authentication Required.")
    if st.button("🔐 Authenticate"):
        try:
            # This attempts to open the browser. 
            # IF IT FAILS: Check your terminal for the URL.
            ee.Authenticate() 
            ee.Initialize(project='stgee-dataset')
            st.session_state.is_auth = True
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}. Check your terminal console for the login link.")
            st.stop()
    st.stop()

check_auth()

# --- STATE MANAGEMENT ---
if 'started' not in st.session_state: st.session_state.started = False
if 'model' not in st.session_state: st.session_state.model = None
if 'best_days' not in st.session_state: st.session_state.best_days = 1
if 'final_vars' not in st.session_state: st.session_state.final_vars = []
if 'df_train' not in st.session_state: st.session_state.df_train = None

# --- SIDEBAR ---
st.sidebar.header("1. Assets")
poly_asset = st.sidebar.text_input("Polygons", "projects/stgee-dataset/assets/export_predictors_polygons2")
pts_asset = st.sidebar.text_input("Points", "projects/stgee-dataset/assets/pointsDate")
pred_asset = st.sidebar.text_input("Prediction Area", "projects/stgee-dataset/assets/export_predictors_polygons2")

st.sidebar.header("2. Parameters")
MIN_D = st.sidebar.number_input("Min Days", 1, 60, 1)
MAX_D = st.sidebar.number_input("Max Days", 1, 60, 30)
FC_DATE = st.sidebar.date_input("Forecast Date", pd.to_datetime("2025-11-26"))
STATIC_VARS = ['Relief_mea', 'S_mean', 'VCv_mean', 'Hill_mean', 'NDVI_mean']

# --- CORE FUNCTIONS ---
@st.cache_data(show_spinner=False)
def get_data(min_d, max_d, _poly, _pts):
    p_fc = ee.FeatureCollection(_poly)
    pt_fc = ee.FeatureCollection(_pts)
    dates = [str(d)[:10] for d in pt_fc.aggregate_array('formatted_date').distinct().getInfo()]
    
    res = []
    bar = st.progress(0, "Downloading...")
    
    for i, date in enumerate(dates):
        try:
            d = ee.Date(date)
            gpm = ee.ImageCollection('JAXA/GPM_L3/GSMaP/v8/operational').select('hourlyPrecipRateGC')
            # Rain accumulation
            bands = [gpm.filterDate(d.advance(-k, 'day'), d).sum().unmask(0).rename(f'Rn{k}') for k in range(min_d, max_d + 1)]
            img = ee.Image.cat(bands)
            
            # Labeling
            daily_pts = pt_fc.filter(ee.Filter.eq('formatted_date', date))
            def set_lbl(f):
                has_ls = daily_pts.filterBounds(f.geometry()).size().gt(0)
                # Numeric ID cleanup
                sid = ee.String(f.get('id')).replace(r'[^0-9]', '', 'g')
                nid = ee.Algorithms.If(sid.length().gt(0), ee.Number.parse(sid), 0)
                return f.set({'P/A': ee.Algorithms.If(has_ls, 1, 0), 'date': date, 'NUM_ID': nid})
            
            stats = img.reduceRegions(p_fc.map(set_lbl), ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True), 1000, tileScale=16)
            df = geemap.ee_to_df(stats)
            
            if not df.empty:
                # Rename cols
                ren = {f'Rn{k}_{s}': f'Rn{k}_{m}' for k in range(min_d, max_d + 1) for s, m in [('mean', 'm'), ('stdDev', 's')]}
                res.append(df.rename(columns=ren))
        except: pass
        bar.progress((i + 1) / len(dates))
    
    bar.empty()
    if not res: return pd.DataFrame()
    
    final = pd.concat(res).sort_values(['date', 'id']).reset_index(drop=True).fillna(0)
    for c in STATIC_VARS: 
        if c not in final.columns: final[c] = 0
    return final

def show_map(df, col, title, asset, pal):
    # Numeric ID for mapping
    df['NID'] = df['id'].apply(lambda x: int(''.join(filter(str.isdigit, str(x))) or 0))
    agg = df.groupby('NID')[col].max().reset_index()
    
    fc = ee.FeatureCollection(asset).map(lambda f: f.set('NID', ee.Number.parse(ee.String(f.get('id')).replace(r'[^0-9]', '', 'g'))))
    img = fc.reduceToImage(['NID'], ee.Reducer.first()).remap(agg['NID'].tolist(), agg[col].tolist()).rename('v')
    
    m = geemap.Map()
    m.centerObject(fc, 10)
    vis = {'min': 0, 'max': 3 if 'Conf' in title else 1, 'palette': pal}
    m.addLayer(img.updateMask(img.gte(0)), vis, title)
    return m

# --- MAIN LAYOUT ---
st.title("PySTGEE: Landslide Hazard")

# LANDING
if not st.session_state.started:
    st.info("Configure assets in sidebar and click Run.")
    if st.button("🚀 Run Analysis", type="primary"):
        st.session_state.started = True
        st.rerun()

# DASHBOARD
else:
    # Base Map
    try:
        m = geemap.Map()
        fc = ee.FeatureCollection(poly_asset)
        m.centerObject(fc, 10)
        m.addLayer(fc.style(color='blue', fillColor='00000000'), {}, "Study Area")
        m.to_streamlit(height=400)
    except: st.error("Invalid Asset")

    t1, t2, t3 = st.tabs(["Calibration", "Validation", "Prediction"])

    with t1:
        if st.button("Start Calibration"):
            with st.spinner("Optimizing..."):
                df = get_data(MIN_D, MAX_D, poly_asset, pts_asset)
                if not df.empty:
                    st.session_state.df_train = df
                    y = df['P/A']
                    best, bw = 0, MIN_D
                    
                    # Find best window
                    for d in range(MIN_D, MAX_D + 1):
                        X = df[[f'Rn{d}_m', f'Rn{d}_s']].fillna(0)
                        rf = RandomForestClassifier(30, max_depth=7, class_weight='balanced', random_state=42).fit(X, y)
                        sc = auc(*roc_curve(y, rf.predict_proba(X)[:, 1])[:2])
                        if sc > best: best, bw = sc, d
                    
                    st.session_state.best_days = bw
                    st.session_state.final_vars = STATIC_VARS + [f'Rn{bw}_m', f'Rn{bw}_s']
                    
                    # Final Model
                    rf_final = RandomForestClassifier(100, max_depth=10, oob_score=True, class_weight='balanced').fit(df[st.session_state.final_vars], y)
                    st.session_state.model = rf_final
                    
                    # Results
                    st.success(f"Best Window: {bw} Days (AUC: {best:.3f})")
                    df['prob'] = rf_final.oob_decision_function_[:, 1]
                    show_map(df, 'prob', 'Calibration', poly_asset, ['white','green','red']).to_streamlit(height=500)

    with t2:
        if st.button("Validate"):
            if st.session_state.model:
                df = st.session_state.df_train
                probs = cross_val_predict(st.session_state.model, df[st.session_state.final_vars], df['P/A'], cv=10, method='predict_proba')[:, 1]
                st.metric("AUC", f"{auc(*roc_curve(df['P/A'], probs)[:2]):.3f}")
                df['v_prob'] = probs
                show_map(df, 'v_prob', 'Validation', poly_asset, ['white','orange','red']).to_streamlit(height=500)
            else: st.error("Calibrate first.")

    with t3:
        if st.button("Predict"):
            if st.session_state.model:
                with st.spinner("Processing..."):
                    td = ee.Date(FC_DATE.strftime('%Y-%m-%d'))
                    gpm = ee.ImageCollection('JAXA/GPM_L3/GSMaP/v8/operational')
                    last = gpm.filterDate('2000-01-01', td.advance(1,'day')).sort('system:time_start', False).first()
                    
                    if last:
                        end_d = ee.Date(last.get('system:time_start'))
                        # Accumulate
                        rn = gpm.filterDate(end_d.advance(-st.session_state.best_days, 'day'), end_d.advance(1,'day')).sum().unmask(0)
                        
                        # Prepare Target Polygons
                        p_fc = ee.FeatureCollection(pred_asset).map(lambda f: f.set('NID', ee.Number.parse(ee.String(f.get('id')).replace(r'[^0-9]', '', 'g'))))
                        stats = geemap.ee_to_df(rn.reduceRegions(p_fc, ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True), 1000))
                        
                        # Map cols
                        bd = st.session_state.best_days
                        stats[f'Rn{bd}_m'] = stats[[c for c in stats.columns if 'mean' in c][0]]
                        stats[f'Rn{bd}_s'] = stats[[c for c in stats.columns if 'stdDev' in c][0]]
                        for c in STATIC_VARS: 
                            if c not in stats.columns: stats[c] = 0
                            
                        stats['risk'] = st.session_state.model.predict_proba(stats[st.session_state.final_vars].fillna(0))[:, 1]
                        st.success(f"Max Risk: {stats['risk'].max():.2f}")
                        show_map(stats, 'risk', 'Prediction', pred_asset, ['green','yellow','red']).to_streamlit(height=600)
            else: st.error("Calibrate first.")
