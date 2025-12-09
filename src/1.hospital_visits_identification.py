# ------------------------------------------------------------------------------
# This script processes synthetic mobile trajectory data used only for demonstration.
# No real or personal data is included. Results generated from this dataset
# SHOULD NOT be interpreted as real-world findings or algorithmic validation.
# ------------------------------------------------------------------------------
 

import pandas as pd 
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import time
import gc

pd.set_option('display.max_columns', None)

starttime_all = time.time()


# ------------------------------------------------------------
# 1. Helper: Memory usage
# ------------------------------------------------------------
def mem_usage(pandas_obj):
    """
    Return memory usage of a pandas object in MB.
    Useful for checking RAM footprint after reading large CSV files.
    """
    if isinstance(pandas_obj, pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else:
        usage_b = pandas_obj.memory_usage(deep=True)
    return "{:03.2f} MB".format(usage_b / 1024 ** 2)


# ------------------------------------------------------------
# 2. Define valid latitude/longitude ranges for each city
# ------------------------------------------------------------
def Filter_lonlat(CITY):
    """
    Return reasonable bounding box for the given CITY.
    This is used to remove impossible coordinates early,
    reducing memory load and computation time.
    """
    if CITY == '北京市':
        return 39.2, 41.8, 115.5, 117.6
    elif CITY == '上海市':
        return 30.4, 32.1, 120.6, 122.5
    elif CITY == '广州市':
        return 22.2, 24.0, 112.7, 114.2
    elif CITY == '深圳市':
        return 22.2, 23.0, 113.5, 114.8
    elif CITY == '成都市':
        return 29.9, 31.6, 102.6, 105.0
    elif CITY == '武汉市':
        return 29.8, 31.2, 113.5, 114.9
    elif CITY == '齐齐哈尔市':
        return 45, 49, 122, 127
    elif CITY == '海东市':
        return 35.2, 37.3, 100.5, 103.3
    elif CITY == '普洱市':
        return 21.9, 25, 99, 102.5
    elif CITY == '日喀则市':
        return 26, 33, 81, 91
    elif CITY == '吐鲁番市':
        return 41, 44, 87, 92


# ------------------------------------------------------------
# 3. Data cleaning
# ------------------------------------------------------------
def data_cleaning(df):
    """
    Clean raw track points:
    - remove non-numeric lon/lat
    - convert data types
    - keep points within valid bounding box
    """
    df.columns = ['id', 'lon', 'lat', 'datetime']

    # Remove rows where lon/lat contain characters
    df = df[df['lon'].str.contains('[a-zA-Z]|[\u4e00-\u9fa5]', na=False) == False]
    df = df[df['lat'].str.contains('[a-zA-Z]|[\u4e00-\u9fa5]', na=False) == False]

    df['id'] = df['id'].astype('category')
    df['lon'] = df['lon'].astype('float32')
    df['lat'] = df['lat'].astype('float32')

    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = Filter_lonlat(CITY)
    df = df[(df['lon'] > LON_MIN) & (df['lon'] < LON_MAX) &
            (df['lat'] > LAT_MIN) & (df['lat'] < LAT_MAX)]

    return df


# ------------------------------------------------------------
# 4. Spatial join: keep points falling inside hospital polygons
# ------------------------------------------------------------
def filter_points_in_hospitals(chunk, polygonShp):
    """
    Convert track points to GeoDataFrame and spatially match
    them to hospital polygons.
    """
    chunk = data_cleaning(chunk)

    geometry = [Point(lon, lat) for lon, lat in zip(chunk['lon'], chunk['lat'])]
    gdf_point = gpd.GeoDataFrame(chunk, geometry=geometry, crs=polygonShp.crs)

    gdf_match = gpd.sjoin(gdf_point, polygonShp, how="inner", predicate="within")[
        ['id', 'lon', 'lat', 'datetime', 'name']
    ]

    return gdf_match


# ------------------------------------------------------------
# 5. Stage 1: Process CSV 
# ------------------------------------------------------------
def preprocess_data(input_csv, polygonShp, chunksize=2_000_000):
    """
    Read large CSV file in chunks and keep only points that fall inside hospitals.
    """

    all_filtered = []
    data_iter = pd.read_csv(
        input_csv, usecols=['id', 'lng', 'lat', 'datetime'],
        dtype={'id': str, 'lng': str, 'lat': str},
        chunksize=chunksize, encoding='utf-8'
    )

    n = 1
    for chunk in data_iter:
        print(f"Chunk {n}: raw size = {mem_usage(chunk)}")
        filtered = filter_points_in_hospitals(chunk, polygonShp)
        print(f" → matched hospital points = {len(filtered)}")
        all_filtered.append(filtered)
        n += 1

    return pd.concat(all_filtered, axis=0)


# ------------------------------------------------------------
# 6. Stage 2: Identify potential single-day hospital visits
# ------------------------------------------------------------
def detect_patient_visits(df):
    """
    Filter users based on time duration inside hospital areas.
    """
    users = df["id"].unique()
    results = []

    for uid in users:
        sub = df[df["id"] == uid].copy()

        min_sec = 30 * 60
        max_sec = 10 * 3600

        sub['datetime'] = pd.to_datetime(sub['datetime'])
        sub = sub.sort_values('datetime')

        start_time = sub['datetime'].min()
        end_time = sub['datetime'].max()
        duration = (end_time - start_time).total_seconds()

        if (start_time.hour >= 7) and (end_time.hour < 19):
            if min_sec <= duration <= max_sec:
                sub['start_time'] = start_time
                sub['end_time'] = end_time
                sub['duration_min'] = duration / 60
                sub = sub.drop_duplicates(subset=['id'])
                results.append(sub)

        del sub
        gc.collect()

    return pd.concat(results, axis=0)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == '__main__':

    CITY_LIST = ['深圳市']

    for CITY in CITY_LIST:

        # Hospital polygon
        PATH_SHP = "../data/AOI_raw/Shenzhen_hospital_AOI.shp"
        polygonShp = gpd.read_file(PATH_SHP)[['name', 'area', 'geometry']]

        # Sample CSV
        INPUT_CSV = "../data/data.csv"

        # Output folder
        OUTPUT_DIR = Path("../data/result_test")
        OUTPUT_DIR.mkdir(exist_ok=True)

        print("=== Stage 1: Filtering points inside hospitals ===")
        df_points = preprocess_data(INPUT_CSV, polygonShp, chunksize=2_000_000)
        df_points.to_csv(OUTPUT_DIR / f"hospital_points_0401.csv", index=False, encoding="utf-8-sig")
        # Note: Since the analysis is performed on synthetic trajectories, the results should not be considered indicative of the method’s empirical performance.
    
        print("=== Stage 2: Detecting single-day hospital visits ===")
        df_patients = detect_patient_visits(df_points)
        df_patients.to_csv(OUTPUT_DIR / f"single_day_patients_0401.csv", index=False, encoding="utf-8-sig")
        # Note: Since the analysis is performed on synthetic trajectories, the results should not be considered indicative of the method’s empirical performance.

        print("Processing finished.")


