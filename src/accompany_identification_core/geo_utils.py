# -*- coding: utf-8 -*-
"""
Geographic utilities for distance calculation and spatial indexing.

Provides Haversine distance calculations and BallTree spatial indexing for
efficient spatial queries on geographic coordinates.

Functions:
- haversine_distance: Calculate distance between two points
- haversine_distance_batch: Batch distance calculations
- build_ball_tree: Build spatial index for fast queries
- find_nearby_points: Find points within radius
- find_nearby_pairs: Find point pairs meeting distance/time thresholds
- deduplicate_pairs: Remove duplicate patient pairs

Constants:
- EARTH_RADIUS_METERS: Average Earth radius in meters
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
        >>> dist = haversine_distance(39.9, 116.4, 31.2, 121.5)
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
        >>> point = (39.9, 116.4)
        >>> cities = [(31.2, 121.5), (22.5, 114.1)]
        >>> distances = haversine_distance_batch(point, cities)
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

    Example:
        >>> # Radians corresponding to 50 meters
        >>> rad = to_radians(50)
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

    Example:
        >>> lats = np.array([39.9, 31.2, 22.5])
        >>> lons = np.array([116.4, 121.5, 114.1])
        >>> tree = build_ball_tree(lats, lons)
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

    Example:
        >>> tree = build_ball_tree(lats, lons)
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

    Example:
        >>> ids = np.array(['A', 'B', 'C', 'D'])
        >>> lats = np.array([39.90, 39.91, 39.92, 31.20])
        >>> lons = np.array([116.40, 116.41, 116.42, 121.50])
        >>> times = np.array(['2024-01-01 10:00:00', '2024-01-01 10:15:00',
        ...                   '2024-01-01 10:05:00', '2024-01-01 10:00:00'],
        ...                  dtype='datetime64')
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
