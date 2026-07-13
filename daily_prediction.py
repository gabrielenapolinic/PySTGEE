#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 PySTGEE: Automated Spatio-Temporal Prediction Pipeline (ULTRA-FAST VERSION)
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

# ------------------------------------------------------------------------------
# 1. CONFIGURATION & ENVIRONMENT SETUP
# ------------------------------------------------------------------------------
EE_PROJECT = 'stgee-dataset'
CATEGORICAL_METRICS = ['LULCmajor', 'Litho']

MODEL_PATH = 'MASTER_MODEL_Japan_fixedLithoRF_U_Kii_fixedLithoRF_U_lsdJapan6690_final.joblib'
STATIC_PRED_CSV = 'Kii_fixedLithoRF_U.gpkg_PRED_static.csv'
BASE_GPKG_PATH = 'Kii_fixedLithoRF_U.gpkg'

OUTPUT_DIR = 'daily_maps'

# ------------------------------------------------------------------------------
# 2. EARTH ENGINE AUTHENTICATION
# ------------------------------------------------------------------------------
def authenticate_gee():
    print("[SYSTEM] Initializing Earth Engine Authentication...")
    key_content = os.environ.get('EE_PRIVATE_KEY')
    if not key_content:
        raise ValueError("CRITICAL ERROR: 'EE_PRIVATE_KEY' not found in environment.")
    
    try:
        service_account_info = json.loads(key_content)
        credentials = ee.ServiceAccountCredentials(service_account_info['client_email'], key_data=key_content)
        ee.Initialize(credentials, project=EE_PROJECT)
        print(f"[SYSTEM] Connected to Earth Engine (Project: {EE_PROJECT}).")
    except Exception as e:
        raise RuntimeError(f"Earth Engine Authentication Failed: {str(e)}")

# ------------------------------------------------------------------------------
# 3. CORE PROCESSING FUNCTIONS (OPTIMIZED)
# ------------------------------------------------------------------------------
def extract_coordinates(uid):
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

def get_rainfall_image(target_date_str, days, source='JAXA'):
    d_target = ee.Date(target_date_str)
    if source == 'ECMWF':
        dataset = ee.ImageCollection("ECMWF/NRT_FORECAST/IFS/OPER").select("total_precipitation_rate_sfc")
        col = dataset.filterDate(d_target.advance(-days, 'day'), d_target.advance(1, 'day'))
        precip_mm_h = col.map(lambda img: img.multiply(3600).rename('precip').copyProperties(img, img.propertyNames()))
        img = ee.Image(ee.Algorithms.If(precip_mm_h.size().gt(0), precip_mm_h.sum(), ee.Image(0)))
        return img.unmask(0).rename(f'Rn{days}_m').toFloat()
    else:
        dataset = ee.ImageCollection("JAXA/GPM_L3/GSMaP/v8/operational").select('hourlyPrecipRateGC')
        col = dataset.filterDate(d_target.advance(-days, 'day'), d_target)
        img = ee.Image(ee.Algorithms.If(col.size().gt(0), col.sum(), ee.Image(0)))
        return img.unmask(0).rename(f'Rn{days}_m').toFloat()

# OTTIMIZZAZIONE 1: Chunk size a 5000 e calcolo coordinate veloce con list comprehension
def extract_rainfall_for_polygons(polygons_df, target_date_str, days, source='JAXA', chunk_size=5000):
    rain_col = f'Rn{days}_m'
    print(f"[GEE] Building {source} image for {days} days ending on {target_date_str}...")
    rain_img = get_rainfall_image(target_date_str, days, source=source)

    df_coords = polygons_df[['poly_uid']].drop_duplicates().copy()
    
    coords_list = [extract_coordinates(u) for u in df_coords['poly_uid']]
    df_coords['lon'] = [c[0] for c in coords_list]
    df_coords['lat'] = [c[1] for c in coords_list]

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
    print(f"[GEE] Extraction successful. Precipitation bounds: {merged[rain_col].min():.2f} mm to {merged[rain_col].max():.2f} mm.")
    return merged

def predict_spacetime(target_date_str, static_df, model, original_predictors, dummies_map, best_days):
    print(f"\n[MODEL] Executing Spatio-Temporal Prediction for {target_date_str}")
    df = static_df.copy()

    cat_cols = [c for c in CATEGORICAL_METRICS if c in original_predictors]
    X_static, _, _ = encode_categoricals(df[original_predictors], original_predictors, cat_cols, dummies_map=dummies_map)
    X_static = X_static.fillna(0)
    
    print("[MODEL] Calculating Base Susceptibility Probabilities...")
    probs = model.predict_proba(X_static)[:, 1]
    df['Susceptibility_Prob'] = probs

    target_dt = datetime.datetime.strptime(target_date_str, '%Y-%m-%d').date()
    is_future = target_dt >= datetime.date.today()
    source = 'ECMWF' if is_future else 'JAXA'

    df_with_rain = extract_rainfall_for_polygons(df, target_date_str, best_days, source=source)

    rain_col = f'Rn{best_days}_m'
    train_ref_rain = 200.0
    print("[MODEL] Fusing temporal dynamics into final susceptibility index...")
    df_with_rain['Final_Dynamic_Susceptibility'] = 1.0 - (1.0 - df_with_rain['Susceptibility_Prob']) * np.exp(-df_with_rain[rain_col] / train_ref_rain)
    df_with_rain.rename(columns={rain_col: 'Rn_m'}, inplace=True)

    return df_with_rain[['poly_uid', 'Susceptibility_Prob', 'Rn_m', 'Final_Dynamic_Susceptibility']]

# OTTIMIZZAZIONE 2: Sostituito Fiona con pyogrio vettorializzato (100 volte più veloce)
def export_prediction_to_geojson(result_df, base_gpkg_path, output_path):
    print(f"[EXPORT] Fast vectorized reconstruction from {base_gpkg_path}...")
    
    try:
        gdf_base = gpd.read_file(base_gpkg_path, engine='pyogrio')
    except Exception:
        gdf_base = gpd.read_file(base_gpkg_path)
    
    gdf_base['geometry'] = gdf_base['geometry'].buffer(0)
    
    rep_points = gdf_base['geometry'].representative_point()
    gdf_base['poly_uid'] = [f"{round(x, 6)}_{round(y, 6)}" for x, y in zip(rep_points.x, rep_points.y)]
    
    df_filtered = result_df.copy()
    df_filtered['poly_uid'] = df_filtered['poly_uid'].astype(str)
    
    merged_gdf = gdf_base[['poly_uid', 'geometry']].merge(df_filtered, on='poly_uid', how='inner')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_gdf.to_file(output_path, driver="GeoJSON")
    print(f"[EXPORT] Successfully wrote {len(merged_gdf)} records to {output_path}")

# ------------------------------------------------------------------------------
# MAIN EXECUTION ROUTINE
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        authenticate_gee()
        target_date = datetime.date.today().strftime('%Y-%m-%d')
        print(f"\n==================================================")
        print(f" COMMENCING DAILY PREDICTION RUN: {target_date}")
        print(f"==================================================\n")
        
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

        final_results = predict_spacetime(
            target_date_str=target_date,
            static_df=prediction_df,
            model=model,
            original_predictors=original_predictors,
            dummies_map=dummies_map,
            best_days=best_days
        )
        
        output_geojson = os.path.join(OUTPUT_DIR, f"prediction_{target_date}.geojson")
        export_prediction_to_geojson(final_results, BASE_GPKG_PATH, output_geojson)
        
        print("\n[SYSTEM] Daily workflow concluded successfully.")
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Pipeline Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
