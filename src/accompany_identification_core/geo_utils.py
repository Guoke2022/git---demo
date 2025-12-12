# -*- coding: utf-8 -*-
"""
Geographic Calculation Tools Module

This module provides geographic distance calculation and spatial indexing functionality based on spherical geometry,
serving as the foundational toolkit for companion patient analysis.
Mainly used for efficient spatio-temporal contact detection and nearest neighbor queries.

Core Features:
==============
1. Spherical Distance Calculation (Haversine Distance)
   - Calculates great-circle distance between two points on Earth's surface using Haversine formula
   - Supports single-point and batch calculations

2. Spatial Indexing (BallTree)
   - Builds BallTree spatial index based on spherical distance
   - Supports fast range queries with time complexity O(log n)
   - Suitable for nearest neighbor retrieval of large-scale trajectory data

3. Nearest Neighbor Query (Nearby Query)
   - Finds all points within specified radius
   - Finds point pairs meeting distance and time thresholds
   - Supports spatio-temporal dual filtering

4. Data Deduplication
   - Deduplicates and normalizes patient pairs
   - Ensures AID ≤ BID uniqueness constraint
   - Retains earliest contact records

Usage Examples:
===============
>>> import numpy as np
>>> from core.geo_utils import *
>>>
>>> # Calculate distance between two points
>>> dist = haversine_distance(39.9, 116.4, 31.2, 121.5)
>>> print(f"Beijing to Shanghai distance: {dist/1000:.1f} km")
>>>
>>> # Build spatial index
>>> lats = np.array([39.9, 31.2, 22.5])
>>> lons = np.array([116.4, 121.5, 114.1])
>>> tree = build_ball_tree(lats, lons)
>>>
>>> # Find points within 50km radius
>>> indices = find_nearby_points(tree, 39.9, 116.4, 50000)
>>>
>>> # Find point pairs meeting thresholds
>>> ids = np.array(['A', 'B', 'C'])
>>> pairs = find_nearby_pairs(ids, lats, lons, None, 100000)

Constant Definitions:
=====================
EARTH_RADIUS_METERS: Earth's average radius (meters), used for distance-radian conversion
"""

from typing import List, Tuple, Optional
import numpy as np
from sklearn.neighbors import BallTree
from haversine import haversine, haversine_vector

# Earth's average radius constant (meters)
EARTH_RADIUS_METERS = 6_371_000


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate Haversine distance between two points (meters)

    Parameters:
        lat1 (float): Latitude of first point (degrees)
        lon1 (float): Longitude of first point (degrees)
        lat2 (float): Latitude of second point (degrees)
        lon2 (float): Longitude of second point (degrees)

    Returns:
        float: Spherical distance between two points (meters)

    Example:
        >>> # Calculate distance from Beijing to Shanghai
        >>> dist = haversine_distance(39.9, 116.4, 31.2, 121.5)
        >>> print(f"{dist/1000:.1f} km")  # Approximately 1067.5 km
    """
    return haversine((lat1, lon1), (lat2, lon2)) * 1000


def haversine_distance_batch(point: Tuple[float, float], points: List[Tuple[float, float]]) -> np.ndarray:
    """
    Batch calculate distances from one point to multiple points (meters)

    Uses vectorized computation to simultaneously calculate Haversine distances from one point to multiple target points,
    more efficient than looping calls to haversine_distance.

    Parameters:
        point (Tuple[float, float]): Source point coordinates (latitude, longitude)
        points (List[Tuple[float, float]]): List of target point coordinates [(latitude, longitude), ...]

    Returns:
        np.ndarray: Distance array (meters), same length as points

    Example:
        >>> point = (39.9, 116.4)  # Beijing
        >>> cities = [(31.2, 121.5), (22.5, 114.1)]  # Shanghai, Shenzhen
        >>> distances = haversine_distance_batch(point, cities)
        >>> print(distances / 1000)  # Convert to kilometers
    """
    return haversine_vector([point] * len(points), points) * 1000


def to_radians(meters: float) -> float:
    """
    Convert distance (meters) to radians

    Used to convert metric distance to radians for BallTree queries, as BallTree uses radians as
    distance measurement unit.

    Parameters:
        meters (float): Distance (meters)

    Returns:
        float: Corresponding radians

    Mathematical Principle:
        Radians = Distance / Earth's Radius
        where Earth's Radius = 6,371,000 meters

    Example:
        >>> # Radians corresponding to 50 meters
        >>> rad = to_radians(50)
        >>> print(f"{rad:.9f} radians")
    """
    return meters / EARTH_RADIUS_METERS


def build_ball_tree(lats: np.ndarray, lons: np.ndarray) -> BallTree:
    """
    Build BallTree spatial index

    Constructs a BallTree spatial index based on coordinate arrays for fast range queries and nearest neighbor searches.
    Uses Haversine metric, suitable for spherical geometry.

    Parameters:
        lats (np.ndarray): Latitude array (degrees)
        lons (np.ndarray): Longitude array (degrees)

    Returns:
        BallTree: Constructed spatial index object

    Data Structure:
        BallTree is a spatial partitioning tree that recursively divides points into hyperspherical regions.
        Queries can prune non-intersecting subtrees with O(log n) time complexity.

    Notes:
        - Input coordinates are in degrees, automatically converted to radians internally
        - lats and lons must have the same length

    Example:
        >>> lats = np.array([39.9, 31.2, 22.5])
        >>> lons = np.array([116.4, 121.5, 114.1])
        >>> tree = build_ball_tree(lats, lons)
        >>> # tree can be used for subsequent queries
    """
    coords = np.column_stack((lats, lons))
    return BallTree(np.radians(coords), metric='haversine')


def find_nearby_points(tree: BallTree, lat: float, lon: float, radius_meters: float) -> np.ndarray:
    """
    Find all point indices within specified radius

    Uses a pre-built BallTree index to find all points within a specified radius around a given point.

    Parameters:
        tree (BallTree): Pre-built BallTree spatial index
        lat (float): Query point latitude (degrees)
        lon (float): Query point longitude (degrees)
        radius_meters (float): Query radius (meters)

    Returns:
        np.ndarray: Array of point indices that meet the condition

    Time Complexity:
        O(log n), where n is the number of points in the index

    Example:
        >>> tree = build_ball_tree(lats, lons)
        >>> # Find all points within 50km of Beijing
        >>> indices = find_nearby_points(tree, 39.9, 116.4, 50000)
        >>> nearby_cities = cities[indices]
    """
    radius_rad = to_radians(radius_meters)
    indices = tree.query_radius(np.radians([[lat, lon]]), r=radius_rad, return_distance=False)[0]
    return indices


def find_nearby_pairs(
    ids: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    times: Optional[np.ndarray],
    distance_threshold: float,
    time_threshold_seconds: Optional[float] = None,
    include_time: bool = True
) -> List[Tuple]:
    """
    Find nearby point pairs that satisfy distance and time thresholds

    This function is the core algorithm for contact detection, using BallTree spatial indexing and time filtering
    to efficiently identify all point pairs that meet spatio-temporal threshold conditions.

    Parameters:
        ids (np.ndarray): Array of point IDs
        lats (np.ndarray): Latitude array (degrees)
        lons (np.ndarray): Longitude array (degrees)
        times (Optional[np.ndarray]): Timestamp array (numpy.datetime64 type), if None, no time filtering is applied
        distance_threshold (float): Distance threshold (meters), distance between two points must be less than or equal to this value
        time_threshold_seconds (Optional[float]): Time threshold (seconds), time difference between two points must be less than or equal to this value
        include_time (bool): Whether to include timestamp fields in results, default True

    Returns:
        List[Tuple]: List of point pairs that meet conditions, each element is a tuple, format depends on include_time:
            - When include_time=True and times is not None:
              (id1, id2, lon1, lat1, lon2, lat2, distance, time1, time2)
            - When include_time=False or times is None:
              (id1, id2, lon1, lat1, lon2, lat2, distance)

    Algorithm Flow:
        1. Build BallTree spatial index
        2. For each point i, query all points within distance threshold
        3. Filter: exclude self (same id) and already processed points (j <= i)
        4. Batch calculate precise Haversine distances
        5. Apply time threshold filtering (if provided)
        6. Collect all point pairs that meet conditions

    Example:
        >>> ids = np.array(['A', 'B', 'C', 'D'])
        >>> lats = np.array([39.90, 39.91, 39.92, 31.20])
        >>> lons = np.array([116.40, 116.41, 116.42, 121.50])
        >>> times = np.array(['2024-01-01 10:00:00', '2024-01-01 10:15:00',
        ...                   '2024-01-01 10:05:00', '2024-01-01 10:00:00'],
        ...                  dtype='datetime64')
        >>> # Find point pairs with distance < 200m and time < 20 minutes
        >>> pairs = find_nearby_pairs(ids, lats, lons, times, 200, 1200)
        >>> print(f"Found {len(pairs)} contact pairs")
    """
    n = len(ids)
    pairs = []

    time_strings = None
    if times is not None:
        time_strings = np.array([str(dt)[:19] for dt in times], dtype='U19')

    def make_pair(i: int, j: int, distance: float) -> Optional[Tuple]:
        if times is not None and time_threshold_seconds is not None:
            time_diff = abs((times[j] - times[i]).astype('timedelta64[s]').astype(int))
            if time_diff > time_threshold_seconds:
                return None

        # Convert numpy types to Python native types to avoid multiprocessing serialization issues
        if include_time and time_strings is not None:
            return (
                str(ids[i]), str(ids[j]),
                float(lons[i]), float(lats[i]),
                float(lons[j]), float(lats[j]),
                float(distance),
                str(time_strings[i]), str(time_strings[j])
            )
        else:
            return (
                str(ids[i]), str(ids[j]),
                float(lons[i]), float(lats[i]),
                float(lons[j]), float(lats[j]),
                float(distance)
            )

    tree = build_ball_tree(lats, lons)
    radius_rad = to_radians(distance_threshold)

    for i in range(n):
        indices = tree.query_radius(np.radians([[lats[i], lons[i]]]), r=radius_rad, return_distance=False)[0]
        valid_j = [j for j in indices if j > i and ids[j] != ids[i]]

        if not valid_j:
            continue

        point1 = (lats[i], lons[i])
        points2 = [(lats[j], lons[j]) for j in valid_j]
        distances = haversine_distance_batch(point1, points2)

        for idx, j in enumerate(valid_j):
            if pair := make_pair(i, j, distances[idx]):
                pairs.append(pair)

    return pairs


def deduplicate_pairs(pairs: List[Tuple], schema: List[str]) -> List[Tuple]:
    """
    Deduplicate and normalize point pairs

    This function ensures each patient pair retains only one record and normalizes ID order (AID ≤ BID),
    avoiding duplicate calculations and storage.

    Parameters:
        pairs (List[Tuple]): Original list of point pairs, each element is a tuple
        schema (List[str]): List of data column names defining field names for each position in tuple
            Standard format: ['AID', 'BID', 'AID_longitude', 'AID_latitude',
                     'BID_longitude', 'BID_latitude', 'Distance', ...]

    Returns:
        List[Tuple]: Deduplicated list of point pairs, each tuple format same as input

    Deduplication Strategy:
        1. Normalize ID order: ensure AID ≤ BID (lexicographic), avoid (A,B) and (B,A) duplicates
        2. Sort: sort by (id1, id2), further sort by time if time fields present
        3. Deduplicate: for each unique (id1, id2) pair, keep the first record (earliest time)

    Empty Data Handling:
        If input is an empty list, directly return empty list

    Example:
        >>> pairs = [
        ...     ('B', 'A', 116.4, 39.9, 121.5, 31.2, 1067.5),
        ...     ('A', 'B', 121.5, 31.2, 116.4, 39.9, 1067.5),  # duplicate
        ...     ('A', 'C', 116.4, 39.9, 114.1, 22.5, 2000.0),
        ... ]
        >>> schema = ['AID', 'BID', 'AID_longitude', 'AID_latitude',
        ...           'BID_longitude', 'BID_latitude', 'Distance']
        >>> deduped = deduplicate_pairs(pairs, schema)
        >>> print(len(deduped))  # 2, removed (B,A) duplicate
        >>> print(deduped[0][0] <= deduped[0][1])  # True, AID <= BID

    """
    import polars as pl

    # Fast return for empty data
    if not pairs:
        return []

    # Convert to Polars DataFrame
    df = pl.DataFrame(data=pairs, orient="row", schema=schema)

    # Normalize ID order: create id1 (smaller ID) and id2 (larger ID) columns
    df = df.with_columns([
        pl.when(pl.col('AID') <= pl.col('BID'))
        .then(pl.col('AID')).otherwise(pl.col('BID')).alias('id1'),
        pl.when(pl.col('AID') <= pl.col('BID'))
        .then(pl.col('BID')).otherwise(pl.col('AID')).alias('id2')
    ])

    # Build sort columns: sort by ID pairs first, further sort by time if time fields present
    sort_cols = ['id1', 'id2']
    if 'AID_start_time' in schema and 'BID_start_time' in schema:
        sort_cols.extend(['AID_start_time', 'BID_start_time'])

    # Sort, deduplicate, rename
    df = df.sort(sort_cols)
    df = df.unique(subset=['id1', 'id2'], keep='first')
    df = df.select(['id1', 'id2'] + schema[2:])
    df = df.rename({'id1': 'AID', 'id2': 'BID'})

    return df.rows()
