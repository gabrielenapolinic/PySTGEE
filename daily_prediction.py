#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 PySTGEE: Automated Spatio-Temporal Landslide Prediction Pipeline
================================================================================
 Headless script designed for GitHub Actions automation.
 Automatically computes landslide susceptibility for TOMORROW by combining:
 1. Static morphological covariates (from pre-computed CSV/GPKG)
 2. Dynamic forecast precipitation (from ECMWF via Google Earth Engine)
 
 Outputs:
 - Ultra-compressed GeoJSON (.geojson.gz) for GIS analysis.
 - Standalone Interactive HTML Web Map (.html) with inspection panels.
================================================================================
"""

import os
import sys
import json
import time
import datetime
import gzip
import shutil
import joblib
import ee
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
from branca.colormap import LinearColormap

# ------------------------------------------------------------------------------
# 1. CONFIGURATION & ENVIRONMENT SETUP
# ------------------------------------------------------------------------------
EE_PROJECT = 'stgee-dataset'
CATEGORICAL_METRICS = ['LULCmajor', 'Litho']

# File paths within the repository
MODEL_PATH = 'MASTER_MODEL_Japan_fixedLithoRF_U_Kii_fixedLithoRF_U_lsdJapan6690_final.joblib'
STATIC_PRED_CSV = 'Kii_fixedLithoRF_U.gpkg_PRED_static.csv'
BASE_GPKG_PATH = 'Kii_fixedLithoRF_U.gpkg'

OUTPUT_DIR = 'daily_maps'

# ------------------------------------------------------------------------------
# 2. EARTH ENGINE AUTHENTICATION
# ------------------------------------------------------------------------------
def authenticate_gee():
    """Authenticates to Google Earth Engine using GitHub Secrets."""
    print("[SYSTEM] Initializing Earth Engine Authentication...")
    key_content = os.environ.get('EE_PRIVATE_KEY')
    
    if not key_content:
        raise ValueError("CRITICAL ERROR: 'EE_PRIVATE_KEY' environment variable not found.")
    
    try:
        service_account_info = json.loads(key_content)
        credentials = ee.ServiceAccountCredentials(service_account_info['client_email'], key_data=key_content)
        ee.Initialize(credentials, project=EE_PROJECT)
        print(f"[SYSTEM] Successfully connected to Earth Engine (Project: {EE_PROJECT}).")
    except Exception as e:
        raise RuntimeError(f"Earth Engine Authentication Failed: {str(e)}")

# ------------------------------------------------------------------------------
# 3. CORE PROCESSING FUNCTIONS (ULTRA-FAST & VECTORIZED)
# ------------------------------------------------------------------------------
def extract_coordinates(uid):
    """Decodes spatial 'poly_uid' string into numeric (longitude, latitude)."""
    uid_str = str(uid)
    if '_' in uid_str:
        parts = uid_str.split('_')
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            return float(parts[0]) / 1e7, float(parts[1]) / 1e7
    else:
        return int(uid_str) / 1e7, 0.0

def encode_categoricals(df, predictor_cols, cat_cols, dummies_map=None):
    """Applies strict One-Hot Encoding matching the calibration model schema."""
    df = df.copy()
    if not cat_cols:
        return df[predictor_cols], predictor_cols, None

    if dummies_map is not None:
        for col, cats in dummies_map.items():
            if col not in df.columns:
                df[col] = 0
            for cat in cats:
                df[f"{col}_{cat}"] = (df[col] == cat).astype(int)
            df.drop(columns=[col], inplace=True, errors='ignore')
        
        all_dummies = [f"{col}_{cat}" for col, cats in dummies_map.items() for cat in cats]
        for c in all_dummies:
            if c not in df.columns:
                df[c] = 0
        new_preds = [c for c in predictor_cols if c not in cat_cols] + all_dummies
        return df[new_preds], new_preds, dummies_map
    return df, predictor_cols, None

def get_rainfall_image(target_date_str, days, source='ECMWF'):
    """Builds GEE cumulative precipitation image over the rolling window."""
    d_target = ee.Date(target_date_str)
    
    if source == 'ECMWF':
        # Forecast data for future prediction (Tomorrow)
        dataset = ee.ImageCollection("ECMWF/NRT_FORECAST/IFS/OPER").select("total_precipitation_rate_sfc")
        col = dataset.filterDate(d_target.advance(-days, 'day'), d_target.advance(1, 'day'))
        # Convert m/s to mm/hour (multiply by 3600)
        precip_mm_h = col.map(lambda img: img.multiply(3600).rename('precip').copyProperties(img, img.propertyNames()))
        img = ee.Image(ee.Algorithms.If(precip_mm_h.size().gt(0), precip_mm_h.sum(), ee.Image(0)))
        return img.unmask(0).rename(f'Rn{days}_m').toFloat()
    else:
        # Historical tracking via JAXA GSMaP
        dataset = ee.ImageCollection("JAXA/GPM_L3/GSMaP/v8/operational").select('hourlyPrecipRateGC')
        col = dataset.filterDate(d_target.advance(-days, 'day'), d_target)
        img = ee.Image(ee.Algorithms.If(col.size().gt(0), col.sum(), ee.Image(0)))
        return img.unmask(0).rename(f'Rn{days}_m').toFloat()

def extract_rainfall_for_polygons(polygons_df, target_date_str, days, source='ECMWF', chunk_size=5000):
    """Performs chunked spatial reduction on GEE to extract rainfall at polygon centroids."""
    rain_col = f'Rn{days}_m'
    print(f"[GEE] Building {source} precipitation image for {days} days ending on {target_date_str}...")
    rain_img = get_rainfall_image(target_date_str, days, source=source)

    df_coords = polygons_df[['poly_uid']].drop_duplicates().copy()
    
    # Fast coordinate extraction via list comprehension
    coords_list = [extract_coordinates(u) for u in df_coords['poly_uid']]
    df_coords['lon'] = [c[0] for c in coords_list]
    df_coords['lat'] = [c[1] for c in coords_list]

    # Pre-allocate dictionary with 0.0 to guarantee safety against GEE failures
    rain_dict = {str(row['poly_uid']): 0.0 for _, row in df_coords.iterrows()}
    features_data = [{'uid': str(u), 'lon': float(lon), 'lat': float(lat)} 
                     for u, lon, lat in zip(df_coords['poly_uid'], df_coords['lon'], df_coords['lat'])]

    total = len(features_data)
    print(f"[GEE] Initiating spatial reduction for {total} points (chunk size: {chunk_size})...")

    for i in range(0, total, chunk_size):
        chunk_data = features_data[i:i+chunk_size]
        ee_features = [ee.Feature(ee.Geometry.Point([d['lon'], d['lat']]), {'poly_uid': d['uid']}) for d in chunk_data]
        fc_chunk = ee.FeatureCollection(ee_features)

        for attempt in range(3):
            try:
                result = rain_img.reduceRegions(
                    collection=fc_chunk,
                    reducer=ee.Reducer.mean(),
                    scale=2000,
                    tileScale=4
                ).getInfo()

                for f in result.get('features', []):
                    props = f.get('properties', {})
                    uid = props.get('poly_uid')
                    # Handle dynamic GEE single-band renaming ('mean' or 'first')
                    val = props.get(rain_col) or props.get('mean') or props.get('first')
                    if uid is not None and val is not None:
                        rain_dict[str(uid)] = float(val)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"[!] GEE API Error on chunk {i//chunk_size + 1}: {str(e)}. Defaulting to 0.0 mm.")
                else:
                    time.sleep(2 ** attempt)

    rain_df = pd.DataFrame(list(rain_dict.items()), columns=['poly_uid', rain_col])
    polygons_df = polygons_df.copy()
    polygons_df['poly_uid'] = polygons_df['poly_uid'].astype(str)
    rain_df['poly_uid'] = rain_df['poly_uid'].astype(str)

    if rain_col in polygons_df.columns:
        polygons_df = polygons_df.drop(columns=[rain_col])

    merged = polygons_df.merge(rain_df, on='poly_uid', how='left')
    merged[rain_col] = merged[rain_col].fillna(0.0)
    print(f"[GEE] Extraction complete. Precipitation bounds: {merged[rain_col].min():.2f} mm to {merged[rain_col].max():.2f} mm.")
    return merged

def predict_spacetime(target_date_str, static_df, model, original_predictors, dummies_map, best_days):
    """Executes the Spatio-Temporal Prediction matrix by combining static and dynamic risk."""
    print(f"\n[MODEL] Executing Spatio-Temporal Prediction for {target_date_str}...")
    df = static_df.copy()

    # Step 1: Base Static Susceptibility
    cat_cols = [c for c in CATEGORICAL_METRICS if c in original_predictors]
    X_static, _, _ = encode_categoricals(df[original_predictors], original_predictors, cat_cols, dummies_map=dummies_map)
    X_static = X_static.fillna(0)
    
    print("[MODEL] Calculating Base Morphological Susceptibility Probabilities...")
    probs = model.predict_proba(X_static)[:, 1]
    df['Susceptibility_Prob'] = probs

    # Step 2: Dynamic Rainfall Extraction (ECMWF Forecast for Tomorrow)
    target_dt = datetime.datetime.strptime(target_date_str, '%Y-%m-%d').date()
    is_future = target_dt >= datetime.date.today()
    source = 'ECMWF' if is_future else 'JAXA'

    df_with_rain = extract_rainfall_for_polygons(df, target_date_str, best_days, source=source)

    # Step 3: Exponential Decay Fusion
    rain_col = f'Rn{best_days}_m'
    train_ref_rain = 200.0
    print("[MODEL] Fusing temporal rainfall dynamics into final susceptibility index...")
    df_with_rain['Final_Dynamic_Susceptibility'] = 1.0 - (1.0 - df_with_rain['Susceptibility_Prob']) * np.exp(-df_with_rain[rain_col] / train_ref_rain)
    df_with_rain.rename(columns={rain_col: 'Rn_m'}, inplace=True)

    return df_with_rain[['poly_uid', 'Susceptibility_Prob', 'Rn_m', 'Final_Dynamic_Susceptibility']]

# ------------------------------------------------------------------------------
# 4. EXPORT PIPELINE: INTERACTIVE HTML DASHBOARD & COMPRESSED GEOJSON
# ------------------------------------------------------------------------------
def export_prediction_to_geojson_and_map(result_df, base_gpkg_path, output_geojson_path, output_html_path, target_date_str):
    """
    Reconstructs spatial topology using pyogrio and generates two outputs:
    1. A Folium Interactive HTML Web Map with hover inspection panels and basemap switcher.
    2. An ultra-compressed .geojson.gz file optimized to stay under GitHub's 100MB limit.
    """
    print(f"[EXPORT] Reconstructing spatial topology from {base_gpkg_path} using pyogrio...")
    
    try:
        gdf_base = gpd.read_file(base_gpkg_path, engine='pyogrio')
    except Exception:
        gdf_base = gpd.read_file(base_gpkg_path)
    
    # Guarantee valid geometries
    gdf_base['geometry'] = gdf_base['geometry'].buffer(0)
    
    rep_points = gdf_base['geometry'].representative_point()
    gdf_base['poly_uid'] = [f"{round(x, 6)}_{round(y, 6)}" for x, y in zip(rep_points.x, rep_points.y)]
    
    df_filtered = result_df.copy()
    df_filtered['poly_uid'] = df_filtered['poly_uid'].astype(str)
    
    essential_cols = ['poly_uid', 'Susceptibility_Prob', 'Rn_m', 'Final_Dynamic_Susceptibility']
    df_to_merge = df_filtered[[c for c in essential_cols if c in df_filtered.columns]]
    
    merged_gdf = gdf_base[['poly_uid', 'geometry']].merge(df_to_merge, on='poly_uid', how='inner')
    
    # --------------------------------------------------------------------------
    # PART A: BUILD INTERACTIVE HTML WEB DASHBOARD WITH PANELS
    # --------------------------------------------------------------------------
    try:
        print("[EXPORT] Building Interactive HTML Web Dashboard with Panels...")
        
        bounds = merged_gdf.total_bounds # [minx, miny, maxx, maxy]
        center_lat = (bounds[1] + bounds[3]) / 2.0
        center_lon = (bounds[0] + bounds[2]) / 2.0
        
        # Initialize map with clean basemap
        m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles="CartoDB positron", control_scale=True)
        
        # Add Esri Satellite Imagery Layer Panel
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite Imagery',
            overlay=False
        ).add_to(m)
        
        # Create Floating Color Legend Panel
        colormap = LinearColormap(
            colors=['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026'],
            vmin=0.0, vmax=1.0,
            caption=f"Dynamic Landslide Susceptibility Index | Forecast Date: {target_date_str}"
        )
        colormap.add_to(m)
        
        # Simplify geometry for fast web browser rendering (~3MB HTML file)
        web_gdf = merged_gdf.copy()
        web_gdf['geometry'] = web_gdf['geometry'].simplify(0.001, preserve_topology=True)
        
        style_function = lambda x: {
            'fillColor': colormap(x['properties']['Final_Dynamic_Susceptibility']),
            'color': 'transparent',
            'weight': 0.2,
            'fillOpacity': 0.75
        }
        
        # Create Hover Inspector Panel (Tooltip)
        tooltip = folium.GeoJsonTooltip(
            fields=['poly_uid', 'Susceptibility_Prob', 'Rn_m', 'Final_Dynamic_Susceptibility'],
            aliases=['Polygon ID:', 'Static Susceptibility:', 'Forecast Rain (mm):', 'Dynamic Risk:'],
            localize=True,
            sticky=False,
            labels=True,
            style="background-color: white; border: 2px solid black; border-radius: 4px; font-family: Arial; font-size: 12px;",
            max_width=400
        )
        
        folium.GeoJson(
            web_gdf,
            name="Landslide Risk Layer",
            style_function=style_function,
            tooltip=tooltip
        ).add_to(m)
        
        # Add Floating Title Dashboard Panel
        title_html = f"""
             <div style="position: fixed; 
                         top: 15px; left: 50px; width: 320px; height: 95px; 
                         z-index:9999; font-size:13px; background-color:white; 
                         border:2px solid #333; border-radius: 6px; padding: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
             <b style="font-size:15px; color:#bd0026;">PySTGEE Automated Monitor</b><br>
             <b>Forecast Date:</b> {target_date_str} (Tomorrow)<br>
             <i style="font-size:11px; color:#555;">Hover over polygons to inspect risk parameters.</i>
             </div>
             """
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Add Layer Control Switcher Panel
        folium.LayerControl(position='topright').add_to(m)
        
        os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
        m.save(output_html_path)
        print(f"[EXPORT] Interactive HTML Dashboard successfully saved to: {output_html_path}")
    except Exception as e_html:
        print(f"[!] Warning: Could not generate HTML Web Map: {str(e_html)}")
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # PART B: ULTRA-COMPRESSED GEOJSON EXPORT (.gz) FOR QGIS/GIS
    # --------------------------------------------------------------------------
    print("[EXPORT] Optimizing geometries and rounding numbers to reduce file size...")
    merged_gdf.geometry = merged_gdf.geometry.set_precision(1e-6)
    for col in ['Susceptibility_Prob', 'Rn_m', 'Final_Dynamic_Susceptibility']:
        if col in merged_gdf.columns:
            merged_gdf[col] = merged_gdf[col].round(4)
    
    temp_geojson = output_geojson_path.replace('.gz', '')
    print(f"[EXPORT] Writing {len(merged_gdf)} records to temporary geometry file...")
    merged_gdf.to_file(temp_geojson, driver="GeoJSON")
    
    print(f"[EXPORT] Compressing to {output_geojson_path} with max GZIP compression (Level 9)...")
    with open(temp_geojson, 'rb') as f_in:
        with gzip.open(output_geojson_path, 'wb', compresslevel=9) as f_out:
            shutil.copyfileobj(f_in, f_out)
            
    # Delete temporary uncompressed file to ensure we stay well below GitHub limits
    os.remove(temp_geojson)
    print(f"[EXPORT] Successfully generated ultra-compressed GeoJSON: {output_geojson_path}!")

# ------------------------------------------------------------------------------
# 5. MAIN EXECUTION ROUTINE (AUTOMATED FOR TOMORROW)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        # 1. Establish Secure GEE Connection
        authenticate_gee()
        
        # 2. Define Execution Temporal Boundary: AUTOMATICALLY FOR TOMORROW
        tomorrow_dt = datetime.date.today() + datetime.timedelta(days=1)
        target_date = tomorrow_dt.strftime('%Y-%m-%d')
        
        print(f"\n==================================================")
        print(f" COMMENCING AUTOMATED PREDICTION FOR TOMORROW: {target_date}")
        print(f"==================================================\n")
        
        # 3. Restore Checkpoint Data (Model & Morphology)
        print("[I/O] Restoring serialized ML Pipeline & Static Features...")
        cached_data = joblib.load(MODEL_PATH)
        model = cached_data['model']
        original_predictors = cached_data.get('original_predictors', [])
        best_days = cached_data.get('best_days', 7)
        dummies_map = cached_data.get('dummies_map', None)
        
        prediction_df = pd.read_csv(STATIC_PRED_CSV, low_memory=False)
        for col in original_predictors:
            if col not in prediction_df.columns:
                prediction_df[col] = 0.0

        # 4. Execute the Spatio-Temporal Prediction Matrix
        final_results = predict_spacetime(
            target_date_str=target_date,
            static_df=prediction_df,
            model=model,
            original_predictors=original_predictors,
            dummies_map=dummies_map,
            best_days=best_days
        )
        
        # 5. Spatialize & Export Dual Results (HTML Dashboard + Compressed GeoJSON)
        output_geojson = os.path.join(OUTPUT_DIR, f"prediction_{target_date}.geojson.gz")
        output_html = os.path.join(OUTPUT_DIR, f"prediction_{target_date}.html")
        
        export_prediction_to_geojson_and_map(final_results, BASE_GPKG_PATH, output_geojson, output_html, target_date)
        
        print("\n[SYSTEM] Automated daily workflow concluded successfully!")
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Pipeline Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
