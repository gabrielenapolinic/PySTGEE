#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 PySTGEE: Automated Spatio-Temporal Prediction Pipeline (ULTRA-FAST VERSION)

 This script automates daily landslide susceptibility predictions using:
 - Pre-trained Random Forest model (loaded from .joblib)
 - Dynamic rainfall data from JAXA GPM (historical) or ECMWF (forecast)
 - Static terrain features (slope, aspect, NDVI, etc.)
 - Output: Interactive HTML map + Compressed GeoJSON for GIS

 Designed to run as a GitHub Actions workflow (daily_run.yml).
================================================================================
"""

import os
import sys
import json
import time
import datetime
import joblib
import ee
import pandas as pd
import geopandas as gpd
import numpy as np

# =============================================================================
# 1. CONFIGURATION & ENVIRONMENT SETUP
# =============================================================================
# Earth Engine project ID (must have access to GPM and ECMWF datasets)
EE_PROJECT = 'stgee-dataset'

# Categorical variables that require one-hot encoding
CATEGORICAL_METRICS = ['LULCmajor', 'Litho']

# Paths to pre-trained model and static data
MODEL_PATH = 'MASTER_MODEL_Japan_fixedLithoRF_U_Kii_fixedLithoRF_U_lsdJapan6690_final.joblib'
STATIC_PRED_CSV = 'Kii_fixedLithoRF_U.gpkg_PRED_static.csv'  # Static features (CSV)
BASE_GPKG_PATH = 'Kii_fixedLithoRF_U.gpkg'                    # Base geometry (GeoPackage)

# Output directory for daily prediction maps
OUTPUT_DIR = 'daily_maps'

# =============================================================================
# 2. EARTH ENGINE AUTHENTICATION
# =============================================================================
def authenticate_gee():
    """
    Authenticates with Google Earth Engine using service account credentials.
    Requires EE_PRIVATE_KEY environment variable (set in GitHub Secrets).
    """
    print("[SYSTEM] Initializing Earth Engine Authentication...")
    key_content = os.environ.get('EE_PRIVATE_KEY')
    if not key_content:
        raise ValueError("CRITICAL ERROR: 'EE_PRIVATE_KEY' not found in environment.")

    try:
        service_account_info = json.loads(key_content)
        credentials = ee.ServiceAccountCredentials(
            service_account_info['client_email'],
            key_data=key_content
        )
        ee.Initialize(credentials, project=EE_PROJECT)
        print(f"[SYSTEM] Connected to Earth Engine (Project: {EE_PROJECT}).")
    except Exception as e:
        raise RuntimeError(f"Earth Engine Authentication Failed: {str(e)}")

# =============================================================================
# 3. CORE PROCESSING FUNCTIONS (OPTIMIZED FOR PERFORMANCE)
# =============================================================================

def extract_coordinates(uid):
    """
    Decodes polygon UID (format: "lon_lat" or "lon_lat_encoded") into (longitude, latitude).
    Handles both raw coordinates and encoded UIDs (divided by 1e7).
    """
    uid_str = str(uid)
    if '_' in uid_str:
        parts = uid_str.split('_')
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            # Fallback for encoded UIDs (e.g., "136108431_34536892" -> lon/1e7, lat/1e7)
            return float(parts[0]) / 1e7, float(parts[1]) / 1e7
    else:
        return int(uid_str) / 1e7, 0.0

def encode_categoricals(df, predictor_cols, cat_cols, dummies_map=None):
    """
    One-hot encodes categorical columns using a pre-computed dummies_map.
    Ensures consistency between training and prediction by adding missing dummy columns.

    Args:
        df: Input DataFrame
        predictor_cols: List of all predictor column names
        cat_cols: List of categorical column names (e.g., ['LULCmajor', 'Litho'])
        dummies_map: Dictionary of {column: [categories]} for one-hot encoding

    Returns:
        Tuple: (encoded_df, new_predictor_cols, dummies_map)
    """
    df = df.copy()
    if not cat_cols:
        return df[predictor_cols], predictor_cols, None

    if dummies_map is not None:
        # Apply pre-computed dummy encoding
        for col, cats in dummies_map.items():
            if col not in df.columns:
                df[col] = 0
            for cat in cats:
                df[f"{col}_{cat}"] = (df[col] == cat).astype(int)
            df.drop(columns=[col], inplace=True, errors='ignore')

        # Ensure all dummy columns exist (for consistency)
        all_dummies = [f"{col}_{cat}" for col, cats in dummies_map.items() for cat in cats]
        for c in all_dummies:
            if c not in df.columns:
                df[c] = 0

        # Update predictor columns list
        new_preds = [c for c in predictor_cols if c not in cat_cols] + all_dummies
        return df[new_preds], new_preds, dummies_map
    return df, predictor_cols, None

def get_rainfall_image(target_date_str, days, source='JAXA'):
    """
    Builds an Earth Engine Image for cumulative rainfall over the specified days.

    Args:
        target_date_str: Date string in 'YYYY-MM-DD' format
        days: Number of days for rainfall accumulation (e.g., 7 or 14)
        source: 'JAXA' (historical, GSMaP) or 'ECMWF' (forecast)

    Returns:
        ee.Image: Cumulative rainfall image (mm) with band name f'Rn{days}_m'
    """
    d_target = ee.Date(target_date_str)

    if source == 'ECMWF':
        # ECMWF Forecast (for future dates)
        dataset = ee.ImageCollection("ECMWF/NRT_FORECAST/IFS/OPER").select("total_precipitation_rate_sfc")
        start = d_target.advance(-days, 'day')
        end = d_target.advance(1, 'day')  # Include the target day
        col = dataset.filterDate(start, end)

        # Convert from mm/s to mm (3600 seconds in an hour)
        precip_mm_h = col.map(
            lambda img: img.multiply(3600).rename('precip').copyProperties(img, img.propertyNames())
        )
        img = ee.Image(ee.Algorithms.If(
            precip_mm_h.size().gt(0),
            precip_mm_h.sum(),
            ee.Image(0)
        ))
        return img.unmask(0).rename(f'Rn{days}_m').toFloat()

    else:
        # JAXA GPM GSMaP (for historical dates)
        dataset = ee.ImageCollection("JAXA/GPM_L3/GSMaP/v8/operational").select('hourlyPrecipRateGC')
        start = d_target.advance(-days, 'day')
        end = d_target  # Exclude the target day (historical only)
        col = dataset.filterDate(start, end)

        img = ee.Image(ee.Algorithms.If(
            col.size().gt(0),
            col.sum(),
            ee.Image(0)
        ))
        return img.unmask(0).rename(f'Rn{days}_m').toFloat()

def extract_rainfall_for_polygons(polygons_df, target_date_str, days, source='JAXA', chunk_size=5000):
    """
    Extracts cumulative rainfall values for each polygon centroid using GEE reduceRegions.
    Processes in chunks to avoid GEE API limits and memory issues.

    Args:
        polygons_df: DataFrame with 'poly_uid' column (polygon identifiers)
        target_date_str: Target date for prediction ('YYYY-MM-DD')
        days: Number of days for rainfall accumulation
        source: 'JAXA' or 'ECMWF'
        chunk_size: Number of polygons to process per batch (default: 5000)

    Returns:
        DataFrame: Original polygons_df with added rainfall column (f'Rn{days}_m')
    """
    rain_col = f'Rn{days}_m'
    print(f"[GEE] Building {source} image for {days} days ending on {target_date_str}...")
    rain_img = get_rainfall_image(target_date_str, days, source=source)

    # Extract unique polygon UIDs and their coordinates
    df_coords = polygons_df[['poly_uid']].drop_duplicates().copy()
    coords_list = [extract_coordinates(u) for u in df_coords['poly_uid']]
    df_coords['lon'] = [c[0] for c in coords_list]
    df_coords['lat'] = [c[1] for c in coords_list]

    # Pre-allocate dictionary with 0.0 for all UIDs (avoids KeyErrors)
    rain_dict = {str(row['poly_uid']): 0.0 for _, row in df_coords.iterrows()}
    features_data = [
        {'uid': str(u), 'lon': float(lon), 'lat': float(lat)}
        for u, lon, lat in zip(df_coords['poly_uid'], df_coords['lon'], df_coords['lat'])
    ]

    total = len(features_data)
    print(f"[GEE] Initiating spatial reduction for {total} points (chunk size: {chunk_size})...")

    # Process in chunks to avoid GEE API timeouts
    for i in range(0, total, chunk_size):
        chunk_data = features_data[i:i+chunk_size]
        ee_features = [
            ee.Feature(ee.Geometry.Point([d['lon'], d['lat']]), {'poly_uid': d['uid']})
            for d in chunk_data
        ]
        fc_chunk = ee.FeatureCollection(ee_features)

        # Retry logic for GEE API stability
        for attempt in range(3):
            try:
                result = rain_img.reduceRegions(
                    collection=fc_chunk,
                    reducer=ee.Reducer.mean(),
                    scale=2000,  # Spatial resolution in meters
                    tileScale=4   # Higher tileScale for large areas
                ).getInfo()

                # Parse results and update rain_dict
                for f in result.get('features', []):
                    props = f.get('properties', {})
                    uid = props.get('poly_uid')
                    # Handle different reducer output names ('RnX_m', 'mean', or 'first')
                    val = props.get(rain_col) or props.get('mean') or props.get('first')
                    if uid is not None and val is not None:
                        rain_dict[str(uid)] = float(val)
                break  # Success: exit retry loop
            except Exception as e:
                if attempt == 2:
                    print(f"[!] GEE API Error on chunk {i//chunk_size + 1}: {str(e)}. Defaulting to 0.0 mm.")
                else:
                    time.sleep(2 ** attempt)  # Exponential backoff

    # Convert rain_dict to DataFrame and merge with input
    rain_df = pd.DataFrame(list(rain_dict.items()), columns=['poly_uid', rain_col])
    polygons_df = polygons_df.copy()
    polygons_df['poly_uid'] = polygons_df['poly_uid'].astype(str)
    rain_df['poly_uid'] = rain_df['poly_uid'].astype(str)

    if rain_col in polygons_df.columns:
        polygons_df = polygons_df.drop(columns=[rain_col])

    merged = polygons_df.merge(rain_df, on='poly_uid', how='left')
    merged[rain_col] = merged[rain_col].fillna(0.0)
    print(f"[GEE] Extraction successful. Precipitation bounds: {merged[rain_col].min():.2f} mm to {merged[rain_col].max():.2f} mm.")
    return merged

def predict_spacetime(target_date_str, static_df, model, original_predictors, dummies_map, best_days):
    """
    Core prediction function: Combines static susceptibility with dynamic rainfall.

    Implements the STGEE formula:
        Final_Susceptibility = 1 - (1 - Static_Prob) * exp(-Rainfall / Reference_Rainfall)

    Args:
        target_date_str: Prediction date ('YYYY-MM-DD')
        static_df: DataFrame with static features (slope, aspect, etc.)
        model: Pre-trained scikit-learn model (Pipeline with RandomForest)
        original_predictors: List of static predictor column names
        dummies_map: Pre-computed dummy encoding map for categoricals
        best_days: Optimal rainfall accumulation window (e.g., 7 or 14 days)

    Returns:
        DataFrame: Predictions with columns ['poly_uid', 'Susceptibility_Prob', 'Rn_m', 'Final_Dynamic_Susceptibility']
    """
    print(f"\n[MODEL] Executing Spatio-Temporal Prediction for {target_date_str}")
    df = static_df.copy()

    # Step 1: Encode categoricals and predict static susceptibility
    cat_cols = [c for c in CATEGORICAL_METRICS if c in original_predictors]
    X_static, _, _ = encode_categoricals(
        df[original_predictors],
        original_predictors,
        cat_cols,
        dummies_map=dummies_map
    )
    X_static = X_static.fillna(0)

    print("[MODEL] Calculating Base Susceptibility Probabilities...")
    probs = model.predict_proba(X_static)[:, 1]  # Probability of class 1 (landslide)
    df['Susceptibility_Prob'] = probs

    # Step 2: Determine rainfall source (ECMWF for future, JAXA for historical)
    target_dt = datetime.datetime.strptime(target_date_str, '%Y-%m-%d').date()
    is_future = target_dt >= datetime.date.today()
    source = 'ECMWF' if is_future else 'JAXA'

    # Extract rainfall for the target date
    df_with_rain = extract_rainfall_for_polygons(
        df, target_date_str, best_days, source=source
    )

    # Step 3: Apply STGEE dynamic susceptibility formula
    rain_col = f'Rn{best_days}_m'
    train_ref_rain = 200.0  # Reference rainfall (mm) from training calibration
    print("[MODEL] Fusing temporal dynamics into final susceptibility index...")
    df_with_rain['Final_Dynamic_Susceptibility'] = (
        1.0 - (1.0 - df_with_rain['Susceptibility_Prob']) *
        np.exp(-df_with_rain[rain_col] / train_ref_rain)
    )
    df_with_rain.rename(columns={rain_col: 'Rn_m'}, inplace=True)

    return df_with_rain[['poly_uid', 'Susceptibility_Prob', 'Rn_m', 'Final_Dynamic_Susceptibility']]

# =============================================================================
# 4. GEOJSON & INTERACTIVE HTML WEB MAP EXPORT PIPELINE
# =============================================================================
def export_prediction_to_geojson_and_map(result_df, base_gpkg_path, output_geojson_path, output_html_path, target_date):
    """
    Exports prediction results to:
    - Interactive HTML map (Folium) for web visualization
    - Compressed GeoJSON (.gz) for GIS tools (QGIS, ArcGIS)

    Args:
        result_df: DataFrame with prediction results (from predict_spacetime)
        base_gpkg_path: Path to base geometry (GeoPackage)
        output_geojson_path: Output path for compressed GeoJSON (.geojson.gz)
        output_html_path: Output path for HTML map
        target_date: Date string for map title
    """
    import gzip
    import shutil
    import folium
    from branca.colormap import LinearColormap

    print(f"[EXPORT] Fast vectorized reconstruction from {base_gpkg_path}...")

    # Load base geometry (polygons)
    try:
        gdf_base = gpd.read_file(base_gpkg_path, engine='pyogrio')
    except Exception:
        gdf_base = gpd.read_file(base_gpkg_path)

    # Ensure valid geometries
    gdf_base['geometry'] = gdf_base['geometry'].buffer(0)

    # Generate poly_uid from representative points (for matching with predictions)
    rep_points = gdf_base['geometry'].representative_point()
    gdf_base['poly_uid'] = [f"{round(x, 6)}_{round(y, 6)}" for x, y in zip(rep_points.x, rep_points.y)]

    # Prepare prediction DataFrame
    df_filtered = result_df.copy()
    df_filtered['poly_uid'] = df_filtered['poly_uid'].astype(str)
    essential_cols = ['poly_uid', 'Susceptibility_Prob', 'Rn_m', 'Final_Dynamic_Susceptibility']
    df_to_merge = df_filtered[[c for c in essential_cols if c in df_filtered.columns]]

    # Merge predictions with geometry
    merged_gdf = gdf_base[['poly_uid', 'geometry']].merge(
        df_to_merge,
        on='poly_uid',
        how='inner'
    )

    # --------------------------------------------------------------------------
    # PART A: BUILD INTERACTIVE HTML WEB MAP WITH PANELS
    # --------------------------------------------------------------------------
    try:
        print("[EXPORT] Building Interactive HTML Web Dashboard with Panels...")

        # 1. Calculate map center from bounding box
        bounds = merged_gdf.total_bounds  # [minx, miny, maxx, maxy]
        center_lat = (bounds[1] + bounds[3]) / 2.0
        center_lon = (bounds[0] + bounds[2]) / 2.0

        # 2. Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=9,
            tiles="CartoDB positron",  # Clean light basemap
            control_scale=True
        )

        # Add Esri Satellite Imagery as an optional layer
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite Imagery',
            overlay=False
        ).add_to(m)

        # 3. Create color legend for susceptibility index
        colormap = LinearColormap(
            colors=['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026'],  # Yellow to dark red
            vmin=0.0,
            vmax=1.0,
            caption=f"Dynamic Landslide Susceptibility Index | Date: {target_date}"
        )
        colormap.add_to(m)

        # 4. Simplify geometries for web performance
        web_gdf = merged_gdf.copy()
        web_gdf['geometry'] = web_gdf['geometry'].simplify(0.0005, preserve_topology=True)

        # Define style function for polygons
        style_function = lambda x: {
            'fillColor': colormap(x['properties']['Final_Dynamic_Susceptibility']),
            'color': 'transparent',
            'weight': 0.3,
            'fillOpacity': 0.75
        }

        # 5. Create hover tooltip for polygon inspection
        tooltip = folium.GeoJsonTooltip(
            fields=['poly_uid', 'Susceptibility_Prob', 'Rn_m', 'Final_Dynamic_Susceptibility'],
            aliases=['Polygon ID:', 'Static Susceptibility:', 'Rainfall (mm):', 'Dynamic Risk:'],
            localize=True,
            sticky=False,
            labels=True,
            style="""
                background-color: #F0EFEF;
                border: 2px solid black;
                border-radius: 4px;
                box-shadow: 3px;
                font-family: Arial; font-size: 12px;
            """,
            max_width=400
        )

        # Add GeoJSON layer to map
        folium.GeoJson(
            web_gdf,
            name="Landslide Risk Layer",
            style_function=style_function,
            tooltip=tooltip
        ).add_to(m)

        # 6. Add custom title panel (floating HTML box)
        title_html = f"""
             <div style="position: fixed;
                         top: 15px; left: 50px; width: 300px; height: 90px;
                         z-index:9999; font-size:13px; background-color:white;
                         border:2px solid #333; border-radius: 6px; padding: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
             <b style="font-size:15px; color:#bd0026;">PySTGEE Web Monitor</b><br>
             <b>Target Date:</b> {target_date}<br>
             <i style="font-size:11px; color:#555;">Hover over polygons to inspect features.</i>
             </div>
             """
        m.get_root().html.add_child(folium.Element(title_html))

        # 7. Add layer switcher panel
        folium.LayerControl(position='topright').add_to(m)

        # Save HTML map
        os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
        m.save(output_html_path)
        print(f"[EXPORT] Interactive HTML map saved successfully: {output_html_path}")
    except Exception as e_html:
        print(f"[!] Warning: Could not generate HTML map: {str(e_html)}")

    # --------------------------------------------------------------------------
    # PART B: ULTRA-COMPRESSED GEOJSON EXPORT (.gz) FOR QGIS/GIS
    # --------------------------------------------------------------------------
    # Optimize geometry precision for file size
    merged_gdf.geometry = merged_gdf.geometry.set_precision(1e-6)

    # Round numeric columns to 4 decimal places
    for col in ['Susceptibility_Prob', 'Rn_m', 'Final_Dynamic_Susceptibility']:
        if col in merged_gdf.columns:
            merged_gdf[col] = merged_gdf[col].round(4)

    # Save temporary GeoJSON, then compress to .gz
    temp_geojson = output_geojson_path.replace('.gz', '')
    print(f"[EXPORT] Writing {len(merged_gdf)} records to temporary geometry file...")
    merged_gdf.to_file(temp_geojson, driver="GeoJSON")

    print(f"[EXPORT] Compressing to {output_geojson_path} with max GZIP compression...")
    with open(temp_geojson, 'rb') as f_in:
        with gzip.open(output_geojson_path, 'wb', compresslevel=9) as f_out:
            shutil.copyfileobj(f_in, f_out)

    os.remove(temp_geojson)
    print(f"[EXPORT] Successfully generated ultra-compressed GeoJSON: {output_geojson_path}!")

# =============================================================================
# MAIN EXECUTION ROUTINE
# =============================================================================
if __name__ == "__main__":
    try:
        # Step 1: Authenticate with Google Earth Engine
        authenticate_gee()

        # Step 2: Set target date (CURRENTLY PREDICTS FOR TODAY - NEEDS TO BE TOMORROW)
        target_date = datetime.date.today().strftime('%Y-%m-%d')
        print(f"\n==================================================")
        print(f" COMMENCING DAILY PREDICTION RUN: {target_date}")
        print(f"==================================================\n")

        # Step 3: Load pre-trained model and static features
        print("[I/O] Restoring serialized ML Pipeline & Static Features...")
        cached_data = joblib.load(MODEL_PATH)
        model = cached_data['model']
        original_predictors = cached_data.get('original_predictors', [])
        best_days = cached_data.get('best_days', 7)
        dummies_map = cached_data.get('dummies_map', None)

        # Load static prediction features (CSV)
        prediction_df = pd.read_csv(STATIC_PRED_CSV, low_memory=False)
        for col in original_predictors:
            if col not in prediction_df.columns:
                prediction_df[col] = 0.0

        # Step 4: Run spatio-temporal prediction
        final_results = predict_spacetime(
            target_date_str=target_date,
            static_df=prediction_df,
            model=model,
            original_predictors=original_predictors,
            dummies_map=dummies_map,
            best_days=best_days
        )

        # Step 5: Define output paths
        output_geojson = os.path.join(OUTPUT_DIR, f"prediction_{target_date}.geojson.gz")
        output_html = os.path.join(OUTPUT_DIR, f"prediction_{target_date}.html")

        # Step 6: Export results
        export_prediction_to_geojson_and_map(
            final_results,
            BASE_GPKG_PATH,
            output_geojson,
            output_html,
            target_date
        )

        print("\n[SYSTEM] Daily workflow concluded successfully.")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Pipeline Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
