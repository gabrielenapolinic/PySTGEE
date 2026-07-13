#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
 PySTGEE: Automated Spatio-Temporal Landslide Prediction Pipeline
================================================================================
 Executes autonomously on GitHub Actions.
 Predictions are automatically generated for 'Tomorrow'.
 
 Dependencies:
 - Trained Random Forest Pipeline (model.joblib)
 - Static morphology CSV
================================================================================
"""

import os
import sys
import json
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
# CONFIGURATION
# ------------------------------------------------------------------------------
EE_PROJECT = 'stgee-dataset'
CATEGORICAL_METRICS = ['LULCmajor', 'Litho']

# Paths (Ensure these match your repository structure)
MODEL_PATH = 'MASTER_MODEL_Japan_fixedLithoRF_U_Kii_fixedLithoRF_U_lsdJapan6690_final.joblib'
STATIC_PRED_CSV = 'Kii_fixedLithoRF_U.gpkg_PRED_static.csv'
BASE_GPKG_PATH = 'Kii_fixedLithoRF_U.gpkg'
OUTPUT_DIR = 'daily_maps'
VIS_PALETTE = ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026']

# ------------------------------------------------------------------------------
# AUTHENTICATION
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
        print(f"[SYSTEM] Successfully connected to Earth Engine.")
    except Exception as e:
        raise RuntimeError(f"Earth Engine Authentication Failed: {str(e)}")

# ------------------------------------------------------------------------------
# PREDICTION ENGINE
# ------------------------------------------------------------------------------
def encode_categoricals(df, predictor_cols, dummies_map=None):
    """Encodes categorical data to match the training pipeline."""
    df = df.copy()
    if dummies_map is not None:
        for col, cats in dummies_map.items():
            if col in df.columns:
                for cat in cats:
                    df[f"{col}_{cat}"] = (df[col] == cat).astype(int)
                df.drop(columns=[col], inplace=True)
        return df, None
    return df, None

def get_rainfall_image(target_date_str, days):
    """Fetches ECMWF forecast precipitation for future dates."""
    d_target = ee.Date(target_date_str)
    dataset = ee.ImageCollection("ECMWF/NRT_FORECAST/IFS/OPER").select("total_precipitation_rate_sfc")
    col = dataset.filterDate(d_target.advance(-days, 'day'), d_target.advance(1, 'day'))
    # Convert m/s to mm/hour
    precip = col.map(lambda img: img.multiply(3600).rename('precip')).sum()
    return precip.unmask(0).rename(f'Rn{days}_m').toFloat()

def predict_spacetime(target_date_str, static_df, model, original_predictors, dummies_map, best_days):
    """Runs the spatio-temporal inference."""
    print(f"[MODEL] Executing Spatio-Temporal Prediction for {target_date_str}")
    df = static_df.copy()
    
    # Encode
    X_static, _ = encode_categoricals(df[original_predictors], original_predictors, dummies_map=dummies_map)
    probs = model.predict_proba(X_static.fillna(0))[:, 1]
    df['Susceptibility_Prob'] = probs
    
    # Rainfall Extraction (ECMWF)
    rain_img = get_rainfall_image(target_date_str, best_days)
    # Placeholder for your specific polygon-point mapping logic
    df['Rn_m'] = 0.0 
    
    df['Final_Dynamic_Susceptibility'] = 1.0 - (1.0 - df['Susceptibility_Prob']) * np.exp(-df['Rn_m'] / 200.0)
    return df[['poly_uid', 'Susceptibility_Prob', 'Rn_m', 'Final_Dynamic_Susceptibility']]

# ------------------------------------------------------------------------------
# EXPORT & VISUALIZATION (HTML + COMPRESSED GEOJSON)
# ------------------------------------------------------------------------------
def export_results(result_df, base_gpkg_path, output_geojson_path, output_html_path, target_date):
    """Generates the GeoJSON (compressed) and the Interactive HTML Dashboard."""
    gdf_base = gpd.read_file(base_gpkg_path)
    # Simplify geometry for web map performance
    gdf_base['geometry'] = gdf_base['geometry'].simplify(0.001)
    
    # Merge prediction results
    merged = gdf_base.merge(result_df, on='poly_uid')
    
    # Export Compressed GeoJSON
    print(f"[EXPORT] Saving compressed GeoJSON to {output_geojson_path}...")
    temp_json = output_geojson_path.replace('.gz', '')
    merged.to_file(temp_json, driver="GeoJSON")
    with open(temp_json, 'rb') as f_in, gzip.open(output_geojson_path, 'wb', compresslevel=9) as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(temp_json)

    # Export Interactive HTML Map
    print(f"[EXPORT] Generating HTML Dashboard...")
    m = folium.Map(location=[merged.geometry.centroid.y.mean(), merged.geometry.centroid.x.mean()], zoom_start=9)
    colormap = LinearColormap(colors=VIS_PALETTE, vmin=0, vmax=1).add_to(m)
    
    folium.GeoJson(
        merged,
        style_function=lambda x: {'fillColor': colormap(x['properties']['Final_Dynamic_Susceptibility']), 'weight': 0.1},
        tooltip=folium.GeoJsonTooltip(fields=['poly_uid', 'Final_Dynamic_Susceptibility'])
    ).add_to(m)
    m.save(output_html_path)

# ------------------------------------------------------------------------------
# MAIN AUTOMATION LOOP
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        authenticate_gee()
        
        # Determine Tomorrow
        tomorrow = datetime.date.today() + datetime.timedelta(days=1)
        target_date = tomorrow.strftime('%Y-%m-%d')
        
        # Load Model Pipeline
        cached_data = joblib.load(MODEL_PATH)
        
        # Load static morphology
        df_base = pd.read_csv(STATIC_PRED_CSV, low_memory=False)
        
        # --- ROBUSTNESS GUARD: Add missing predictors if necessary ---
        original_preds = cached_data.get('original_predictors', [])
        for col in original_preds:
            if col not in df_base.columns:
                print(f"[WARNING] Missing predictor '{col}' detected. Filling with 0.0.")
                df_base[col] = 0.0
        
        # Run prediction
        results = predict_spacetime(
            target_date, 
            df_base, 
            cached_data['model'], 
            original_preds, 
            cached_data.get('dummies_map'), 
            cached_data.get('best_days', 7)
        )
        
        # Define paths
        out_gz = os.path.join(OUTPUT_DIR, f"prediction_{target_date}.geojson.gz")
        out_html = os.path.join(OUTPUT_DIR, f"prediction_{target_date}.html")
        
        # Export
        export_results(results, BASE_GPKG_PATH, out_gz, out_html, target_date)
        
        # Create "latest" aliases (The zero-touch master link logic)
        shutil.copy(out_gz, os.path.join(OUTPUT_DIR, "latest_map.geojson.gz"))
        shutil.copy(out_html, os.path.join(OUTPUT_DIR, "latest_map.html"))
        
        print(f"[SUCCESS] Pipeline finished for {target_date}.")
        
    except Exception as e:
        print(f"[CRITICAL] Pipeline Failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
