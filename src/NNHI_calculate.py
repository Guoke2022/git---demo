"""
===============================================================
NNHI_calculate.py
===============================================================

This script computes the Nearest N Hospitals Index (NNHI) and the corresponding
road-network distances based on a set of home grid points and hospital locations.

The script merges two components:
1. NNHI computation
2. Road-network distance computation using OSMnx

This file is organized for academic publication and replication guidance.
It does not require the actual dataset or city-specific road networks to run.


---------------------------------------------------------------------------
Input Data Requirements
---------------------------------------------------------------------------

1. Home–hospital matching dataset: pandas DataFrame with:
   - 'hosp_id'        : str, hospital identifier representing the hospital
                        actually visited by the individual
   - 'grid_lon_hou'   : float, home-grid longitude (GCJ02)
   - 'grid_lat_hou'   : float, home-grid latitude (GCJ02)

2. Hospital dataset: pandas DataFrame with:
   - 'hosp_id'         : str, unique hospital identifier
   - 'POINT_X_GCJ02'   : float, hospital longitude (GCJ02)
   - 'POINT_Y_GCJ02'   : float, hospital latitude (GCJ02)

3. Road network:
   A {CITY}.graphml file that can be loaded via OSMnx:
       ox.load_graphml("path/to/{CITY}.graphml")


---------------------------------------------------------------------------
Output Data Format
---------------------------------------------------------------------------

A pandas DataFrame containing:

  Home-grid information:
   - 'grid_lon_hou'
   - 'grid_lat_hou'

  NNHI results:
   - 'NNHI'                   : rank (1,2,3,…) of the hospital actually visited
                               among all hospitals sorted by straight-line distance

  Nearest hospitals (straight-line):
   - 'closest_hosp',  'closest_distance'   (km)
   - 'second_hosp',   'second_distance'    (km)
   - 'third_hosp',    'third_distance'     (km)

  Road-network distances (km):
   - 'road_closest'   : road distance to the nearest hospital
   - 'road_second'    : road distance to the second-nearest hospital
   - 'road_third'     : road distance to the third-nearest hospital
   - 'road_actual'    : road distance from home-grid to the hospital actually visited
                        (based on 'hosp_id')

---------------------------------------------------------------------------
Notes
---------------------------------------------------------------------------
• The GCJ02→WGS84 conversion is approximate but sufficiently accurate for routing.  
• Users must supply actual datasets and road-network files for computation.  
"""

import math
import numpy as np
import pandas as pd
import osmnx as ox
from haversine import haversine, Unit


# =============================================================================
# Section 1 — Great-circle (Haversine) distance utilities
# =============================================================================
def haversine_distance(lat1, lon1, lat2, lon2):
    """Return great-circle distance (meters) between two coordinates."""
    return haversine((lat1, lon1), (lat2, lon2), unit=Unit.METERS)


def compute_haversine(point1, point2):
    """Wrapper for direct use with (lng, lat) coordinate tuples."""
    p1 = (point1[1], point1[0])
    p2 = (point2[1], point2[0])
    return haversine(p1, p2, unit=Unit.METERS)


# =============================================================================
# Section 2 — NNHI identification using straight-line distances
# =============================================================================
def find_closest_hospital_info(row, df_hospital):
    """
    Identify the 1st, 2nd, and 3rd nearest hospitals using Haversine distance.

    Parameters
    ----------
    row : DataFrame row
        Includes home-grid coords + the hospital actually used ('hosp_id')
    df_hospital : DataFrame
        All hospitals with coordinates

    Returns
    -------
    tuple
        (NNHI_rank,
         first_hosp, first_lat, first_lon, first_dist_km,
         second_hosp, second_lat, second_lon, second_dist_km,
         third_hosp, third_lat, third_lon, third_dist_km)
    """

    target_hosp = row['hosp_id']

    distances = df_hospital.apply(
        lambda x: haversine_distance(
            row['grid_lat_hou'], row['grid_lon_hou'],
            x['POINT_Y_GCJ02'], x['POINT_X_GCJ02']
        ),
        axis=1
    )

    sorted_idx = distances.nsmallest(len(df_hospital)).index
    hosp_sorted = df_hospital.loc[sorted_idx, 'hosp_id'].values

    # Find rank of actual hospital
    nnhi_rank = next((i + 1 for i, name in enumerate(hosp_sorted) if name == target_hosp), -1)

    def pick(k):
        idx = sorted_idx[k]
        return (
            df_hospital.loc[idx, 'hosp_id'],
            df_hospital.loc[idx, 'POINT_Y_GCJ02'],
            df_hospital.loc[idx, 'POINT_X_GCJ02'],
            distances[idx] / 1000.0
        )

    first = pick(0)
    second = pick(1)
    third = pick(2)

    return (nnhi_rank, *first, *second, *third)


# =============================================================================
# Section 3 — GCJ02 → WGS84 conversion (for China)
# =============================================================================
A = 6378245.0
EE = 6.693421622965943e-3

def GCJ02_to_WGS84(lng, lat):
    """Approximate conversion from GCJ02 to WGS84."""
    if not (73.66 < lng < 135.05 and 3.86 < lat < 53.55):
        return [lng, lat]

    dlat = _transform_lat(lng - 105.0, lat - 35.0)
    dlng = _transform_lng(lng - 105.0, lat - 35.0)
    radlat = lat * math.pi / 180.0
    magic = 1 - EE * math.sin(radlat)**2
    sqrtmagic = math.sqrt(magic)

    dlat = (dlat * 180.0) / ((A * (1 - EE)) / (magic * sqrtmagic) * math.pi)
    dlng = (dlng * 180.0) / (A / sqrtmagic * math.cos(radlat) * math.pi)

    mglat = lat + dlat
    mglng = lng + dlng

    return [lng * 2 - mglng, lat * 2 - mglat]

def _transform_lat(lng, lat):
    ret = (-100.0 + 2.0 * lng + 3.0 * lat +
           0.2 * lat*lat + 0.1 * lng*lat + 0.2*math.sqrt(abs(lng)))
    ret += (20.0*math.sin(6.0*lng*math.pi) +
            20.0*math.sin(2.0*lng*math.pi))*2.0/3.0
    ret += (20.0*math.sin(lat*math.pi) +
            40.0*math.sin(lat/3.0*math.pi))*2.0/3.0
    return ret

def _transform_lng(lng, lat):
    ret = (300.0 + lng + 2.0*lat +
           0.1*lng*lng + 0.1*lng*lat + 0.1*math.sqrt(abs(lng)))
    ret += (20.0*math.sin(6.0*lng*math.pi) +
            20.0*math.sin(2.0*lng*math.pi))*2.0/3.0
    ret += (150.0*math.sin(lng/12.0*math.pi) +
            300.0*math.sin(lng/30.0*math.pi))*2.0/3.0
    return ret


# =============================================================================
# Section 4 — Road-network distance using OSMnx
# =============================================================================
def get_road_distance(start, end, graph, city_name):
    """
    Compute OSM-network-based driving distance.
    Fallback: Haversine distance if route is unavailable.
    """
    try:

        start = GCJ02_to_WGS84(start[0], start[1])
        end = GCJ02_to_WGS84(end[0], end[1])

        from_node = ox.nearest_nodes(graph, start[0], start[1])
        to_node = ox.nearest_nodes(graph, end[0], end[1])

        route = ox.shortest_path(graph, from_node, to_node, weight="length")
        if route is None:
            raise ValueError("No route available")

        edges = ox.graph_to_gdfs(graph.subgraph(route), nodes=False)
        if edges.empty:
            raise ValueError("Empty route")

        return edges["length"].sum()

    except ValueError:
        return compute_haversine(start, end)


# =============================================================================
# Section 5 — Main workflow
# =============================================================================
def compute_NNHI_and_road_distance(CITY, home_data, df_hospital, graph):
    """
    Compute NNHI and road-network distance for each unique home-grid location.

    Parameters
    ----------
    CITY : str
    home_data : DataFrame
        Includes 'hosp_id', 'grid_lon_hou', 'grid_lat_hou'
    df_hospital : DataFrame
    graph : OSMnx road network (loaded separately)

    Returns
    -------
    DataFrame
    """

    # Step 1: Unique home-grid locations
    home_unique = home_data[['hosp_id', 'grid_lon_hou', 'grid_lat_hou']].drop_duplicates()

    # Step 2: NNHI + straight-line nearest hospitals
    nnhi_raw = home_unique.apply(find_closest_hospital_info, axis=1, args=(df_hospital,))
    cols = [
        'NNHI',
        'closest_hosp', 'closest_lat', 'closest_lon', 'closest_distance',
        'second_hosp', 'second_lat', 'second_lon', 'second_distance',
        'third_hosp', 'third_lat', 'third_lon', 'third_distance'
    ]
    df_nnhi = pd.DataFrame(nnhi_raw.tolist(), columns=cols)

    merged = pd.concat([home_unique.reset_index(drop=True), df_nnhi], axis=1)

    # Step 3: Road distances (no multiprocessing here for simplicity)
    road_closest = []
    road_second = []
    road_third = []
    road_actual = []

    for _, row in merged.iterrows():

        # road distance to 1st, 2nd, 3rd nearest hospitals
        road_closest.append(
            get_road_distance(
                (row['grid_lon_hou'], row['grid_lat_hou']),
                (row['closest_lon'], row['closest_lat']),
                graph, CITY
            ) / 1000
        )
        road_second.append(
            get_road_distance(
                (row['grid_lon_hou'], row['grid_lat_hou']),
                (row['second_lon'], row['second_lat']),
                graph, CITY
            ) / 1000
        )
        road_third.append(
            get_road_distance(
                (row['grid_lon_hou'], row['grid_lat_hou']),
                (row['third_lon'], row['third_lat']),
                graph, CITY
            ) / 1000
        )

        # NEW: distance to actual hospital (used in NNHI)
        hosp_row = df_hospital.loc[df_hospital['hosp_id'] == row['hosp_id']].iloc[0]
        road_actual.append(
            get_road_distance(
                (row['grid_lon_hou'], row['grid_lat_hou']),
                (hosp_row['POINT_X_GCJ02'], hosp_row['POINT_Y_GCJ02']),
                graph, CITY
            ) / 1000
        )

    # save results
    merged['road_closest'] = np.round(road_closest, 2)
    merged['road_second'] = np.round(road_second, 2)
    merged['road_third'] = np.round(road_third, 2)
    merged['road_actual'] = np.round(road_actual, 2)

    return merged