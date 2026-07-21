#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
 PySTGEE: Automated Spatio-Temporal Landslide Prediction Pipeline
================================================================================
 Executes autonomously on GitHub Actions.
 Predictions are automatically generated for 'Tomorrow'.
 Features GEE Dynamic Rainfall Extraction (ECMWF/GSMaP), high-fidelity 
 rasterization, and custom UI dashboard panels.
================================================================================
"""

import os
import sys
import json
import time
import datetime
import shutil
import warnings
import gzip
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

# Exact Palette: Green -> White -> Dark Red
VIS_PALETTE = ['#006b0b', '#1b7b25', '#4e9956', '#dbeadd', '#ffffff', '#f0b2ae', '#eb958f', '#df564d', '#d10e00']

def authenticate_gee():
    """Initializes GEE connection using environment secrets."""
    print("[SYSTEM] Authenticating Earth Engine...")
    key_content = os.environ.get('EE_PRIVATE_KEY')
    if not key_content: 
        raise ValueError("CRITICAL ERROR: 'EE_PRIVATE_KEY' not found.")
    service_account_info = json.loads(key_content)
    credentials = ee.ServiceAccountCredentials(service_account_info['client_email'], key_data=key_content)
    ee.Initialize(credentials, project=EE_PROJECT)

def extract_coordinates(uid):
    """Decode poly_uid to (longitude, latitude)."""
    uid_str = str(uid)
    if '_' in uid_str:
        parts = uid_str.split('_')
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            try:
                return float(parts[0]) / 1e7, float(parts[1]) / 1e7
            except:
                return 0.0, 0.0
    else:
        try:
            return int(uid_str) / 1e7, 0.0
        except:
            return 0.0, 0.0

def get_rainfall_image(target_date_str, days, source='JAXA'):
    """Builds the EE Image for cumulative rainfall over the specified days."""
    d_target = ee.Date(target_date_str)
    start = d_target.advance(-days, 'day')
    end = d_target.advance(1, 'day')

    if source == 'ECMWF':
        # Forecast data (Future)
        dataset = ee.ImageCollection("ECMWF/NRT_FORECAST/IFS/OPER").select("total_precipitation_rate_sfc")
        col = dataset.filterDate(start, end)
        # Convert m/s -> mm/h (1 m/s = 3,600,000 mm/h).
        # NOTE: Divided by 12.0 to normalize overlapping forecast runs in Earth Engine
        # preventing artificial 1000+ mm saturation and keeping precipitation realistic.
        precip_mm_h = col.map(lambda img: img.multiply(3600).rename('precip').copyProperties(img, img.propertyNames()))
        img = ee.Image(ee.Algorithms.If(precip_mm_h.size().gt(0), precip_mm_h.sum().divide(12.0), ee.Image(0)))
        return img.unmask(0).rename(f'Rn{days}_m').toFloat()
    else:
        # Historical satellite observations (GSMaP Operational mm/hr)
        dataset = ee.ImageCollection("JAXA/GPM_L3/GSMaP/v8/operational").select('hourlyPrecipRateGC')
        col = dataset.filterDate(start, d_target)
        img = ee.Image(ee.Algorithms.If(col.size().gt(0), col.sum(), ee.Image(0)))
        return img.unmask(0).rename(f'Rn{days}_m').toFloat()

def extract_rainfall_for_polygons(polygons_df, target_date_str, days, source='JAXA', chunk_size=2000):
    """Extracts cumulative rainfall for each polygon centroid using GEE reduceRegions in robust batches."""
    rain_col = f'Rn{days}_m'
    print(f"[GEE] Building {source} rainfall image for {days} days ending on {target_date_str}...")
    rain_img = get_rainfall_image(target_date_str, days, source=source)

    df_coords = polygons_df[['poly_uid']].drop_duplicates().copy()
    df_coords[['lon', 'lat']] = df_coords['poly_uid'].apply(lambda x: pd.Series(extract_coordinates(x)))

    # Pre-allocate dictionary with 0.0 to prevent missing keys
    rain_dict = {str(row['poly_uid']): 0.0 for _, row in df_coords.iterrows()}
    features_data = [{'uid': str(row['poly_uid']), 'lon': float(row['lon']), 'lat': float(row['lat'])} for _, row in df_coords.iterrows()]

    total = len(features_data)
    print(f"[GEE] Extracting dynamic rainfall for {total} centroids in chunks of {chunk_size}...")

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

                features = result.get('features', [])
                if len(features) > 0:
                    for f in features:
                        props = f.get('properties', {})
                        uid = props.get('poly_uid')
                        # Handle GEE single-band reducer renaming
                        val = props.get(rain_col) or props.get('mean') or props.get('first')
                        if uid is not None and val is not None:
                            rain_dict[str(uid)] = float(val)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  [!] Chunk {i//chunk_size + 1} failed: {e}. Kept as 0.0 mm.")
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
    print(f"[GEE] Rainfall extraction complete. Min: {merged[rain_col].min():.2f} mm | Max: {merged[rain_col].max():.2f} mm")
    return merged

def get_prediction_logic(target_date_str, static_df, model, dummies_map, best_days=14):
    """Core ML logic: One-Hot Encoding, Dynamic GEE Rainfall Extraction, and Spatio-Temporal Inference."""
    df = static_df.copy()
    if 'poly_uid' not in df.columns:
        df['poly_uid'] = [f"ID_{i}" for i in range(len(df))]
        
    # 1. Apply One-Hot Encoding
    if dummies_map is not None:
        for col, cats in dummies_map.items():
            if col in df.columns:
                for cat in cats:
                    df[f"{col}_{cat}"] = (df[col] == cat).astype(int)
        raw_cat_cols = list(dummies_map.keys())
        df.drop(columns=raw_cat_cols, inplace=True, errors='ignore')
    
    # 2. Align features perfectly with trained Random Forest
    expected_cols = model.feature_names_in_
    X_static = df.reindex(columns=expected_cols, fill_value=0.0)
    
    # 3. Static Inference
    print("[ML] Predicting static morphological susceptibility...")
    df['Susceptibility_Prob'] = model.predict_proba(X_static)[:, 1]
    
    # 4. Dynamic Rainfall Extraction from GEE
    target_dt = datetime.datetime.strptime(target_date_str, '%Y-%m-%d').date()
    is_future = target_dt >= datetime.date.today()
    source = 'ECMWF' if is_future else 'JAXA'
    
    df_with_rain = extract_rainfall_for_polygons(df, target_date_str, best_days, source=source, chunk_size=2000)
    
    # 5. Spatio-Temporal Combined Hazard Calculation
    print("[ML] Combining static susceptibility with antecedent rainfall accumulation...")
    rain_col = f'Rn{best_days}_m'
    df_with_rain['Final_Dynamic_Susceptibility'] = 1.0 - (1.0 - df_with_rain['Susceptibility_Prob']) * np.exp(-df_with_rain[rain_col] / 200.0)
    df_with_rain.rename(columns={rain_col: 'Rn_m'}, inplace=True)
    
    return df_with_rain[['poly_uid', 'Susceptibility_Prob', 'Rn_m', 'Final_Dynamic_Susceptibility']]

def export_results(result_df, base_gpkg_path, output_geojson_path, output_html_path, target_date):
    """Generates uncompressed GeoJSON and high-fidelity Raster-on-HTML Dashboard."""
    print("[EXPORT] Reading base geometries...")
    gdf_base = gpd.read_file(base_gpkg_path)
    
    print(f"[JOIN] Geometries count: {len(gdf_base)} | Predictions count: {len(result_df)}")
    
    # --- BULLETPROOF JOIN LOGIC ---
    if len(gdf_base) == len(result_df):
        print("[JOIN] Exact row count match! Mapping predictions directly by index to guarantee 100% polygon coverage.")
        merged = gdf_base.copy()
        merged['Final_Dynamic_Susceptibility'] = result_df['Final_Dynamic_Susceptibility'].values
        merged['Susceptibility_Prob'] = result_df['Susceptibility_Prob'].values
        merged['Rn_m'] = result_df['Rn_m'].values if 'Rn_m' in result_df.columns else 0.0
        if 'poly_uid' not in merged.columns:
            merged['poly_uid'] = result_df['poly_uid'].values if 'poly_uid' in result_df.columns else [f"ID_{i}" for i in range(len(merged))]
    else:
        print("[JOIN] Row counts differ. Attempting robust merge on 'poly_uid'...")
        if 'poly_uid' not in gdf_base.columns:
            gdf_base['poly_uid'] = [f"{round(x, 6)}_{round(y, 6)}" for x, y in zip(gdf_base.geometry.centroid.x, gdf_base.geometry.centroid.y)]
        merged = gdf_base.merge(result_df, on='poly_uid', how='inner')

    if len(merged) < 100:
        raise ValueError(f"CRITICAL ERROR: Merged GeoDataFrame has only {len(merged)} rows! Join failed.")
        
    print(f"[SUCCESS] Prepared {len(merged)} polygons for visualization and GIS export.")
        
    if merged.crs is None:
        merged.set_crs("EPSG:4326", inplace=True)
    elif merged.crs != "EPSG:4326":
        merged = merged.to_crs("EPSG:4326")
        
    merged['Final_Dynamic_Susceptibility'] = merged['Final_Dynamic_Susceptibility'].fillna(0.0).astype(float)
    merged['poly_uid'] = merged['poly_uid'].astype(str)
    
    # --- RASTERIZATION FOR WEB MAP ---
    print("[EXPORT] Rasterizing geometries for web visualization...")
    minx, miny, maxx, maxy = merged.total_bounds
    res = 0.0002
    width = int((maxx - minx) / res)
    height = int((maxy - miny) / res)
    max_pixels = 6000
    if width > max_pixels or height > max_pixels:
        scale = max(width, height) / max_pixels
        width = int(width / scale)
        height = int(height / scale)
        
    transform_mat = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)
    
    shapes_for_rasterize = [(geom, val) for geom, val in zip(merged.geometry, merged['Final_Dynamic_Susceptibility']) if geom is not None and not geom.is_empty]
    
    raster = rasterio.features.rasterize(
        shapes=shapes_for_rasterize,
        out_shape=(height, width),
        transform=transform_mat,
        fill=-9999.0,
        dtype=np.float32,
        all_touched=True
    )
    
    masked_raster = np.ma.masked_where(raster == -9999.0, raster)
    
    cmap = mcolors.LinearSegmentedColormap.from_list("pystgee_custom", VIS_PALETTE)
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    
    rgba_array = cmap(norm(masked_raster))
    rgba_array[masked_raster.mask] = [0, 0, 0, 0]
    
    img = Image.fromarray((rgba_array * 255).astype(np.uint8))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # --- BUILD FOLIUM MAP & INJECT DASHBOARD PANELS ---
    print("[EXPORT] Building interactive HTML dashboard with custom UI panels...")
    center_lat = (miny + maxy) / 2.0
    center_lon = (minx + maxx) / 2.0
    m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles="OpenStreetMap")
    
    folium.raster_layers.ImageOverlay(
        image=f"data:image/png;base64,{img_str}",
        bounds=[[(miny), (minx)], [(maxy), (maxx)]],
        name=f"Susceptibility ({target_date})",
        opacity=0.85
    ).add_to(m)
    
    folium.LayerControl(position="topleft").add_to(m)
    
    max_val = merged['Final_Dynamic_Susceptibility'].max()
    mean_val = merged['Final_Dynamic_Susceptibility'].mean()
    poly_count = len(merged)
    mean_rain = merged['Rn_m'].mean() if 'Rn_m' in merged.columns else 0.0
    max_rain = merged['Rn_m'].max() if 'Rn_m' in merged.columns else 0.0
    
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
    
    <div class="pystgee-panel dashboard-panel">
        <div class="pystgee-title">
            <span>PySTGEE Forecast</span>
            <span style="font-size: 12px; background: #e9ecef; padding: 2px 6px; border-radius: 4px;">{target_date}</span>
        </div>
        <div class="stat-row"><span>Mean Rainfall:</span> <span style="font-weight:600;">{mean_rain:.2f} mm</span></div>
        <div class="stat-row"><span>Max Cumulative Rainfall:</span> <span style="font-weight:600;">{max_rain:.2f} mm</span></div>
        <div class="stat-row"><span>Mean Susceptibility:</span> <span style="font-weight:600;">{mean_val:.3f}</span></div>
        <div class="stat-row"><span>Max Susceptibility:</span> <span class="stat-val">{max_val:.3f}</span></div>
        <a href="daily_maps/latest_map.geojson" class="download-btn" download>&#11015; Download GIS Data (.geojson)</a>
    </div>
    
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
    
    m.get_root().html.add_child(folium.Element(ui_html))
    m.save(output_html_path)

    # --- SAVE UNCOMPRESSED GEOJSON ---
    print("[EXPORT] Optimizing and saving uncompressed GeoJSON for GIS users...")
    export_gdf = merged[['poly_uid', 'Susceptibility_Prob', 'Rn_m', 'Final_Dynamic_Susceptibility', 'geometry']].copy()
    export_gdf['geometry'] = export_gdf['geometry'].simplify(0.0001, preserve_topology=True)
    export_gdf.to_file(output_geojson_path, driver="GeoJSON")
    print(f"[SUCCESS] GeoJSON saved uncompressed: {os.path.getsize(output_geojson_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    try:
        authenticate_gee()
        target_date = (datetime.date.today() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        cached_data = joblib.load(MODEL_PATH)
        df_base = pd.read_csv(STATIC_PRED_CSV, low_memory=False)
        
        best_days = cached_data.get('best_days', 14)
        print(f"[SYSTEM] Model based on {best_days}-day antecedent rainfall window.")
        
        results = get_prediction_logic(
            target_date, 
            df_base, 
            cached_data['model'], 
            cached_data.get('dummies_map'),
            best_days=best_days
        )
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_json = os.path.join(OUTPUT_DIR, f"prediction_{target_date}.geojson")
        out_html = os.path.join(OUTPUT_DIR, f"prediction_{target_date}.html")
        
        export_results(results, BASE_GPKG_PATH, out_json, out_html, target_date)
        
        shutil.copy(out_json, os.path.join(OUTPUT_DIR, "latest_map.geojson"))
        shutil.copy(out_html, os.path.join(OUTPUT_DIR, "latest_map.html"))
        shutil.copy(out_html, "index.html")
        
        print("[SUCCESS] Pipeline finished.")
    except Exception as e:
        print(f"[CRITICAL] Pipeline Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
