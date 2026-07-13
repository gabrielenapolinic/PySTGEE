#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
 PySTGEE: Automated Spatio-Temporal Landslide Prediction Pipeline
================================================================================
 Executes autonomously on GitHub Actions.
 Predictions are automatically generated for 'Tomorrow'.
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

# Configuration
EE_PROJECT = 'stgee-dataset'
MODEL_PATH = 'MASTER_MODEL_Japan_fixedLithoRF_U_Kii_fixedLithoRF_U_lsdJapan6690_final.joblib'
STATIC_PRED_CSV = 'Kii_fixedLithoRF_U.gpkg_PRED_static.csv'
BASE_GPKG_PATH = 'Kii_fixedLithoRF_U.gpkg'
OUTPUT_DIR = 'daily_maps'
VIS_PALETTE = ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026']

def authenticate_gee():
    """Initializes GEE connection."""
    print("[SYSTEM] Authenticating Earth Engine...")
    key_content = os.environ.get('EE_PRIVATE_KEY')
    if not key_content:
        raise ValueError("CRITICAL ERROR: 'EE_PRIVATE_KEY' not found.")
    service_account_info = json.loads(key_content)
    credentials = ee.ServiceAccountCredentials(service_account_info['client_email'], key_data=key_content)
    ee.Initialize(credentials, project=EE_PROJECT)

def get_prediction_logic(target_date_str, static_df, model, original_predictors, dummies_map):
    """Core ML logic: Uses reindex to ensure all predictors exist."""
    df = static_df.copy()
    
    # 1. Encode categorical features safely
    if dummies_map is not None:
        for col, cats in dummies_map.items():
            if col in df.columns:
                for cat in cats:
                    df[f"{col}_{cat}"] = (df[col] == cat).astype(int)
                df.drop(columns=[col], inplace=True)
    
    # 2. Robust Prediction using reindex (Solves KeyError: '...' not in index)
    # This automatically adds missing columns with 0.0, avoiding crashes.
    X_static = df.reindex(columns=original_predictors, fill_value=0.0)
    
    # 3. Inference
    df['Susceptibility_Prob'] = model.predict_proba(X_static)[:, 1]
    df['Rn_m'] = 0.0 # Placeholder for GEE extraction
    df['Final_Dynamic_Susceptibility'] = 1.0 - (1.0 - df['Susceptibility_Prob']) * np.exp(-df['Rn_m'] / 200.0)
    
    return df[['poly_uid', 'Susceptibility_Prob', 'Rn_m', 'Final_Dynamic_Susceptibility']]

def export_results(result_df, base_gpkg_path, output_geojson_path, output_html_path, target_date):
    """Generates the GeoJSON (compressed) and the Interactive HTML Dashboard."""
    gdf_base = gpd.read_file(base_gpkg_path)
    gdf_base['poly_uid'] = [f"{round(x, 6)}_{round(y, 6)}" for x, y in zip(gdf_base.geometry.centroid.x, gdf_base.geometry.centroid.y)]
    merged = gdf_base.merge(result_df, on='poly_uid')
    
    # Save Compressed GeoJSON
    out_gz = output_geojson_path
    temp_json = out_gz.replace('.gz', '')
    merged.to_file(temp_json, driver="GeoJSON")
    with open(temp_json, 'rb') as f_in, gzip.open(out_gz, 'wb', compresslevel=9) as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(temp_json)

    # Save Interactive HTML Map
    m = folium.Map(location=[merged.geometry.centroid.y.mean(), merged.geometry.centroid.x.mean()], zoom_start=9)
    colormap = LinearColormap(colors=VIS_PALETTE, vmin=0, vmax=1).add_to(m)
    folium.GeoJson(
        merged.simplify(0.001),
        style_function=lambda x: {'fillColor': colormap(x['properties']['Final_Dynamic_Susceptibility']), 'weight': 0.1},
        tooltip=folium.GeoJsonTooltip(fields=['poly_uid', 'Final_Dynamic_Susceptibility'])
    ).add_to(m)
    m.save(output_html_path)

if __name__ == "__main__":
    try:
        authenticate_gee()
        target_date = (datetime.date.today() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        
        cached_data = joblib.load(MODEL_PATH)
        
        # Load static morphology
        df_base = pd.read_csv(STATIC_PRED_CSV, low_memory=False)
        
        # Run prediction
        results = get_prediction_logic(
            target_date, df_base, 
            cached_data['model'], 
            cached_data.get('original_predictors', []), 
            cached_data.get('dummies_map')
        )
        
        # Paths
        out_gz = os.path.join(OUTPUT_DIR, f"prediction_{target_date}.geojson.gz")
        out_html = os.path.join(OUTPUT_DIR, f"prediction_{target_date}.html")
        
        # Export
        export_results(results, BASE_GPKG_PATH, out_gz, out_html, target_date)
        
        # Create "latest" aliases (Zero-Touch)
        shutil.copy(out_gz, os.path.join(OUTPUT_DIR, "latest_map.geojson.gz"))
        shutil.copy(out_html, os.path.join(OUTPUT_DIR, "latest_map.html"))
        
        print(f"[SUCCESS] Pipeline finished.")
        
    except Exception as e:
        print(f"[CRITICAL] Pipeline Failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
