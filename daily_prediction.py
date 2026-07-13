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
import warnings
import joblib
import ee
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
from branca.colormap import LinearColormap

# Ignore CRS centroid warnings for cleaner logs
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration
EE_PROJECT = 'stgee-dataset'
MODEL_PATH = 'MASTER_MODEL_Japan_fixedLithoRF_U_Kii_fixedLithoRF_U_lsdJapan6690_final.joblib'
STATIC_PRED_CSV = 'Kii_fixedLithoRF_U.gpkg_PRED_static.csv'
BASE_GPKG_PATH = 'Kii_fixedLithoRF_U.gpkg'
OUTPUT_DIR = 'daily_maps'
VIS_PALETTE = ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026']

def authenticate_gee():
    """Initializes GEE connection using environment secrets."""
    print("[SYSTEM] Authenticating Earth Engine...")
    key_content = os.environ.get('EE_PRIVATE_KEY')
    if not key_content: 
        raise ValueError("CRITICAL ERROR: 'EE_PRIVATE_KEY' not found.")
    service_account_info = json.loads(key_content)
    credentials = ee.ServiceAccountCredentials(service_account_info['client_email'], key_data=key_content)
    ee.Initialize(credentials, project=EE_PROJECT)

def get_prediction_logic(target_date_str, static_df, model, dummies_map):
    """Core ML logic: Uses model's internal memory for perfect feature alignment."""
    df = static_df.copy()
    
    # 1. Apply One-Hot Encoding
    if dummies_map is not None:
        for col, cats in dummies_map.items():
            if col in df.columns:
                for cat in cats:
                    df[f"{col}_{cat}"] = (df[col] == cat).astype(int)
        
        # Remove raw categorical columns to avoid confusing the model pipeline
        raw_cat_cols = list(dummies_map.keys())
        df.drop(columns=raw_cat_cols, inplace=True, errors='ignore')
    
    # 2. Extract the EXACT columns the model expects directly from the trained model
    expected_cols = model.feature_names_in_
    
    # 3. Align the DataFrame perfectly (fills missing with 0.0, drops unexpected)
    X_static = df.reindex(columns=expected_cols, fill_value=0.0)
    
    # 4. Inference
    df['Susceptibility_Prob'] = model.predict_proba(X_static)[:, 1]
    df['Rn_m'] = 0.0  # Placeholder for GEE precipitation extraction
    df['Final_Dynamic_Susceptibility'] = 1.0 - (1.0 - df['Susceptibility_Prob']) * np.exp(-df['Rn_m'] / 200.0)
    
    # Ensure ID column exists
    if 'poly_uid' not in df.columns:
        df['poly_uid'] = [f"ID_{i}" for i in range(len(df))]
        
    return df[['poly_uid', 'Susceptibility_Prob', 'Rn_m', 'Final_Dynamic_Susceptibility']]

def export_results(result_df, base_gpkg_path, output_geojson_path, output_html_path, target_date):
    """Generates GeoJSON and Interactive HTML Dashboard with robust join logic."""
    gdf_base = gpd.read_file(base_gpkg_path)
    
    # --- ROBUST & FOOLPROOF JOIN LOGIC ---
    # Instead of fragile coordinate strings, we check for a real shared ID column
    # or align directly by row index since GeoPackage and CSV are the exact same dataset.
    if 'poly_uid' in gdf_base.columns and 'poly_uid' in result_df.columns:
        print("[JOIN] Merging on existing 'poly_uid' column...")
        merged = gdf_base.merge(result_df, on='poly_uid')
    else:
        common_cols = [c for c in gdf_base.columns if c in result_df.columns and c != 'geometry']
        if common_cols:
            print(f"[JOIN] Merging on detected common column: '{common_cols[0]}'")
            merged = gdf_base.merge(result_df, on=common_cols[0])
        elif len(gdf_base) == len(result_df):
            print("[JOIN] Aligning GeoPackage and CSV directly by index (identical row count)...")
            merged = gdf_base.copy()
            for col in ['Susceptibility_Prob', 'Rn_m', 'Final_Dynamic_Susceptibility']:
                merged[col] = result_df[col].values
            if 'poly_uid' not in merged.columns:
                merged['poly_uid'] = result_df['poly_uid'].values if 'poly_uid' in result_df.columns else [f"ID_{i}" for i in range(len(merged))]
        else:
            raise ValueError(f"CRITICAL: Cannot join GeoPackage ({len(gdf_base)} rows) and CSV ({len(result_df)} rows). No common ID column and row counts differ.")

    # --- STRICT GUARD AGAINST SILENT EMPTY MAPS ---
    if len(merged) == 0:
        raise ValueError("CRITICAL ERROR: Merged GeoDataFrame has 0 rows! The join between geometry and predictions failed.")
        
    print(f"[DEBUG] Successfully prepared {len(merged)} polygons for visualization.")
    
    # Ensure valid WGS84 CRS for Leaflet/Folium rendering
    if merged.crs is None:
        merged.set_crs("EPSG:4326", inplace=True)
    elif merged.crs != "EPSG:4326":
        merged = merged.to_crs("EPSG:4326")
    
    # Simplify geometry safely while keeping topology intact
    merged['geometry'] = merged['geometry'].simplify(0.0005, preserve_topology=True)
    
    # Clean data types to prevent JSON serialization errors
    merged['Final_Dynamic_Susceptibility'] = merged['Final_Dynamic_Susceptibility'].fillna(0.0).astype(float)
    merged['poly_uid'] = merged['poly_uid'].astype(str)
    
    # Save Compressed GeoJSON
    print("[EXPORT] Saving compressed GeoJSON...")
    temp_json = output_geojson_path.replace('.gz', '')
    merged.to_file(temp_json, driver="GeoJSON")
    with open(temp_json, 'rb') as f_in, gzip.open(output_geojson_path, 'wb', compresslevel=9) as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(temp_json)

    # Save Interactive HTML Map
    print("[EXPORT] Generating HTML Dashboard...")
    m = folium.Map(location=[merged.geometry.centroid.y.mean(), merged.geometry.centroid.x.mean()], zoom_start=9)
    colormap = LinearColormap(colors=VIS_PALETTE, vmin=0, vmax=1).add_to(m)
    
    folium.GeoJson(
        merged[['poly_uid', 'Final_Dynamic_Susceptibility', 'geometry']],
        style_function=lambda x: {
            'fillColor': colormap(x['properties'].get('Final_Dynamic_Susceptibility', 0.0)), 
            'color': '#333333',
            'weight': 0.2,
            'fillOpacity': 0.75
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['poly_uid', 'Final_Dynamic_Susceptibility'],
            aliases=['Polygon ID:', 'Susceptibility Prob:'],
            localize=True
        )
    ).add_to(m)
    m.save(output_html_path)

if __name__ == "__main__":
    try:
        authenticate_gee()
        target_date = (datetime.date.today() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        cached_data = joblib.load(MODEL_PATH)
        df_base = pd.read_csv(STATIC_PRED_CSV, low_memory=False)
        
        results = get_prediction_logic(
            target_date, 
            df_base, 
            cached_data['model'], 
            cached_data.get('dummies_map')
        )
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_gz = os.path.join(OUTPUT_DIR, f"prediction_{target_date}.geojson.gz")
        out_html = os.path.join(OUTPUT_DIR, f"prediction_{target_date}.html")
        
        export_results(results, BASE_GPKG_PATH, out_gz, out_html, target_date)
        
        # "Latest" Aliases for zero-touch web routing
        shutil.copy(out_gz, os.path.join(OUTPUT_DIR, "latest_map.geojson.gz"))
        shutil.copy(out_html, os.path.join(OUTPUT_DIR, "latest_map.html"))
        
        # Copy directly to root index.html so the website opens immediately on the map
        shutil.copy(out_html, "index.html")
        
        print("[SUCCESS] Pipeline finished.")
    except Exception as e:
        print(f"[CRITICAL] Pipeline Failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
