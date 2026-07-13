#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
 PySTGEE: Automated Spatio-Temporal Landslide Prediction Pipeline
================================================================================
 Executes autonomously on GitHub Actions.
 Predictions are automatically generated for 'Tomorrow'.
 Features high-fidelity rasterization and custom UI dashboard panels.
================================================================================
"""

import os
import sys
import json
import datetime
import gzip
import shutil
import warnings
from io import BytesIO
import base64
import joblib
import ee
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
import rasterio
import rasterio.features
import rasterio.transform
from PIL import Image
import matplotlib.colors as mcolors

# Ignore CRS centroid warnings for cleaner logs
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration
EE_PROJECT = 'stgee-dataset'
MODEL_PATH = 'MASTER_MODEL_Japan_fixedLithoRF_U_Kii_fixedLithoRF_U_lsdJapan6690_final.joblib'
STATIC_PRED_CSV = 'Kii_fixedLithoRF_U.gpkg_PRED_static.csv'
BASE_GPKG_PATH = 'Kii_fixedLithoRF_U.gpkg'
OUTPUT_DIR = 'daily_maps'

# Exact Palette from notebook: Green -> White -> Dark Red
VIS_PALETTE = ['#006b0b', '#1b7b25', '#4e9956', '#dbeadd', '#ffffff', '#f0b2ae', '#eb958f', '#df564d', '#d10e00'][cite: 1]

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
    
    # 1. Apply One-Hot Encoding[cite: 1]
    if dummies_map is not None:
        for col, cats in dummies_map.items():
            if col in df.columns:
                for cat in cats:
                    df[f"{col}_{cat}"] = (df[col] == cat).astype(int)[cite: 1]
        
        # Remove raw categorical columns to avoid confusing the model pipeline[cite: 1]
        raw_cat_cols = list(dummies_map.keys())[cite: 1]
        df.drop(columns=raw_cat_cols, inplace=True, errors='ignore')[cite: 1]
    
    # 2. Extract the EXACT columns the model expects directly from the trained model[cite: 1]
    expected_cols = model.feature_names_in_[cite: 1]
    
    # 3. Align the DataFrame perfectly (fills missing with 0.0, drops unexpected)[cite: 1]
    X_static = df.reindex(columns=expected_cols, fill_value=0.0)[cite: 1]
    
    # 4. Inference[cite: 1]
    df['Susceptibility_Prob'] = model.predict_proba(X_static)[:, 1][cite: 1]
    df['Rn_m'] = 0.0  # Placeholder for GEE precipitation extraction[cite: 1]
    df['Final_Dynamic_Susceptibility'] = 1.0 - (1.0 - df['Susceptibility_Prob']) * np.exp(-df['Rn_m'] / 200.0)[cite: 1]
    
    if 'poly_uid' not in df.columns:
        df['poly_uid'] = [f"ID_{i}" for i in range(len(df))][cite: 1]
        
    return df[['poly_uid', 'Susceptibility_Prob', 'Rn_m', 'Final_Dynamic_Susceptibility']][cite: 1]

def export_results(result_df, base_gpkg_path, output_geojson_path, output_html_path, target_date):
    """Generates compressed GeoJSON and high-fidelity Raster-on-HTML Dashboard[cite: 1]."""
    print("[EXPORT] Reading base geometries...")
    gdf_base = gpd.read_file(base_gpkg_path)[cite: 1]
    
    if 'poly_uid' not in gdf_base.columns:
        gdf_base['poly_uid'] = [f"{round(x, 6)}_{round(y, 6)}" for x, y in zip(gdf_base.geometry.centroid.x, gdf_base.geometry.centroid.y)][cite: 1]
        
    print("[JOIN] Merging predictions with geometries...")
    if 'poly_uid' in gdf_base.columns and 'poly_uid' in result_df.columns:
        merged = gdf_base.merge(result_df, on='poly_uid')[cite: 1]
    else:
        common_cols = [c for c in gdf_base.columns if c in result_df.columns and c != 'geometry'][cite: 1]
        if common_cols:
            merged = gdf_base.merge(result_df, on=common_cols[0])[cite: 1]
        elif len(gdf_base) == len(result_df):
            merged = gdf_base.copy()[cite: 1]
            for col in ['Susceptibility_Prob', 'Rn_m', 'Final_Dynamic_Susceptibility']:
                merged[col] = result_df[col].values[cite: 1]
            if 'poly_uid' not in merged.columns:
                merged['poly_uid'] = result_df['poly_uid'].values if 'poly_uid' in result_df.columns else [f"ID_{i}" for i in range(len(merged))][cite: 1]
        else:
            raise ValueError("Cannot join GeoPackage and CSV.")

    if len(merged) == 0:
        raise ValueError("CRITICAL ERROR: Merged GeoDataFrame has 0 rows!")[cite: 1]
        
    if merged.crs is None:
        merged.set_crs("EPSG:4326", inplace=True)[cite: 1]
    elif merged.crs != "EPSG:4326":
        merged = merged.to_crs("EPSG:4326")[cite: 1]
        
    merged['Final_Dynamic_Susceptibility'] = merged['Final_Dynamic_Susceptibility'].fillna(0.0).astype(float)[cite: 1]
    merged['poly_uid'] = merged['poly_uid'].astype(str)[cite: 1]
    
    # Save Compressed GeoJSON (Full fidelity, NO simplification for GIS users)[cite: 1]
    print("[EXPORT] Saving compressed GeoJSON...")
    temp_json = output_geojson_path.replace('.gz', '')
    merged.to_file(temp_json, driver="GeoJSON")[cite: 1]
    with open(temp_json, 'rb') as f_in, gzip.open(output_geojson_path, 'wb', compresslevel=9) as f_out:
        shutil.copyfileobj(f_in, f_out)[cite: 1]
    os.remove(temp_json)[cite: 1]

    # --- RASTERIZATION FOR WEB MAP (100% Geometry Fidelity, Zero Browser Lag) ---[cite: 1]
    print("[EXPORT] Rasterizing geometries for web visualization...")
    minx, miny, maxx, maxy = merged.total_bounds[cite: 1]
    res = 0.0002  # High resolution grid[cite: 1]
    width = int((maxx - minx) / res)[cite: 1]
    height = int((maxy - miny) / res)[cite: 1]
    max_pixels = 6000  # Optimal size for crisp rendering and fast loading[cite: 1]
    if width > max_pixels or height > max_pixels:
        scale = max(width, height) / max_pixels[cite: 1]
        width = int(width / scale)[cite: 1]
        height = int(height / scale)[cite: 1]
        
    transform_mat = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)[cite: 1]
    
    # Prepare shapes: list of (geometry, value)[cite: 1]
    shapes_for_rasterize = [(geom, val) for geom, val in zip(merged.geometry, merged['Final_Dynamic_Susceptibility']) if geom is not None and not geom.is_empty][cite: 1]
    
    raster = rasterio.features.rasterize(
        shapes=shapes_for_rasterize,
        out_shape=(height, width),
        transform=transform_mat,
        fill=-9999.0,
        dtype=np.float32,
        all_touched=True
    )[cite: 1]
    
    masked_raster = np.ma.masked_where(raster == -9999.0, raster)[cite: 1]
    
    # Exact Palette from original notebook: Green -> White -> Dark Red[cite: 1]
    cmap = mcolors.LinearSegmentedColormap.from_list("pystgee_custom", VIS_PALETTE)[cite: 1]
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)[cite: 1]
    
    rgba_array = cmap(norm(masked_raster))[cite: 1]
    rgba_array[masked_raster.mask] = [0, 0, 0, 0]  # Transparent background[cite: 1]
    
    img = Image.fromarray((rgba_array * 255).astype(np.uint8))[cite: 1]
    buffered = BytesIO()[cite: 1]
    img.save(buffered, format="PNG")[cite: 1]
    img_str = base64.b64encode(buffered.getvalue()).decode()[cite: 1]
    
    # --- BUILD FOLIUM MAP & INJECT DASHBOARD PANELS ---[cite: 1]
    print("[EXPORT] Building interactive HTML dashboard with custom UI panels...")
    center_lat = (miny + maxy) / 2.0[cite: 1]
    center_lon = (minx + maxx) / 2.0[cite: 1]
    m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles="OpenStreetMap")[cite: 1]
    
    # Add Raster Image Overlay[cite: 1]
    folium.raster_layers.ImageOverlay(
        image=f"data:image/png;base64,{img_str}",
        bounds=[[(miny), (minx)], [(maxy), (maxx)]],
        name=f"Susceptibility ({target_date})",
        opacity=0.85
    ).add_to(m)[cite: 1]
    
    folium.LayerControl(position="topleft").add_to(m)[cite: 1]
    
    # Summary Stats for UI Dashboard[cite: 1]
    max_val = merged['Final_Dynamic_Susceptibility'].max()[cite: 1]
    mean_val = merged['Final_Dynamic_Susceptibility'].mean()[cite: 1]
    poly_count = len(merged)[cite: 1]
    rain_active = (merged['Rn_m'] > 0).sum() if 'Rn_m' in merged.columns else 0[cite: 1]
    
    # Inject Custom CSS and UI Panels (Dashboard & Legend) matching notebook style[cite: 1]
    ui_html = f"""
    <style>
        .pystgee-panel {{
            position: fixed;
            z-index: 9999;
            background-color: rgba(255, 255, 255, 0.95);
            border: 1px solid #ccc;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #333 !important;
            padding: 15px;
        }}
        .dashboard-panel {{
            top: 20px;
            right: 20px;
            width: 310px;
        }}
        .legend-panel {{
            bottom: 25px;
            left: 20px;
            width: 280px;
        }}
        .pystgee-title {{
            font-weight: bold;
            font-size: 15px;
            border-bottom: 2px solid #333;
            padding-bottom: 6px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            margin: 6px 0;
            font-size: 13px;
        }}
        .stat-val {{
            font-weight: 600;
            color: #d10e00;
        }}
        .download-btn {{
            display: block;
            width: 100%;
            text-align: center;
            background-color: #28a745;
            color: white !important;
            text-decoration: none;
            font-weight: bold;
            font-size: 13px;
            padding: 8px 0;
            border-radius: 5px;
            margin-top: 12px;
            transition: background-color 0.2s;
        }}
        .download-btn:hover {{
            background-color: #1b7b25;
        }}
        .legend-bar {{
            width: 100%;
            height: 14px;
            background: linear-gradient(to right, {", ".join(VIS_PALETTE)});
            border: 1px solid #999;
            border-radius: 3px;
            margin: 8px 0 4px 0;
        }}
        .legend-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            font-weight: bold;
            color: #555;
        }}
    </style>
    
    <!-- Main Info Dashboard -->
    <div class="pystgee-panel dashboard-panel">
        <div class="pystgee-title">
            <span>PySTGEE Forecast</span>
            <span style="font-size: 12px; background: #e9ecef; padding: 2px 6px; border-radius: 4px;">{target_date}</span>
        </div>
        <div class="stat-row"><span>Monitored Units:</span> <span style="font-weight:600;">{poly_count:,}</span></div>
        <div class="stat-row"><span>Active Rain Zones:</span> <span style="font-weight:600;">{rain_active:,}</span></div>
        <div class="stat-row"><span>Mean Risk Index:</span> <span style="font-weight:600;">{mean_val:.3f}</span></div>
        <div class="stat-row"><span>Max Risk Index:</span> <span class="stat-val">{max_val:.3f}</span></div>
        <a href="daily_maps/latest_map.geojson.gz" class="download-btn" download>&#11015; Download GIS Data (.geojson.gz)</a>
    </div>
    
    <!-- Color Legend Panel -->
    <div class="pystgee-panel legend-panel">
        <div style="font-weight: bold; font-size: 13px; margin-bottom: 4px;">Landslide Susceptibility Index</div>
        <div class="legend-bar"></div>
        <div class="legend-labels">
            <span>0.0 (Low)</span>
            <span>0.5 (Mod)</span>
            <span>1.0 (High)</span>
        </div>
    </div>
    """
    
    m.get_root().html.add_child(folium.Element(ui_html))[cite: 1]
    m.save(output_html_path)[cite: 1]

if __name__ == "__main__":
    try:
        authenticate_gee()
        target_date = (datetime.date.today() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        cached_data = joblib.load(MODEL_PATH)
        df_base = pd.read_csv(STATIC_PRED_CSV, low_memory=False)[cite: 1]
        
        results = get_prediction_logic(
            target_date, 
            df_base, 
            cached_data['model'], 
            cached_data.get('dummies_map')
        )[cite: 1]
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)[cite: 1]
        out_gz = os.path.join(OUTPUT_DIR, f"prediction_{target_date}.geojson.gz")
        out_html = os.path.join(OUTPUT_DIR, f"prediction_{target_date}.html")
        
        export_results(results, BASE_GPKG_PATH, out_gz, out_html, target_date)
        
        # "Latest" Aliases for zero-touch web routing[cite: 1]
        shutil.copy(out_gz, os.path.join(OUTPUT_DIR, "latest_map.geojson.gz"))[cite: 1]
        shutil.copy(out_html, os.path.join(OUTPUT_DIR, "latest_map.html"))[cite: 1]
        
        # Copy directly to root index.html so the website opens immediately on the map[cite: 1]
        shutil.copy(out_html, "index.html")[cite: 1]
        
        print("[SUCCESS] Pipeline finished.")
    except Exception as e:
        print(f"[CRITICAL] Pipeline Failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
