# -*- coding: utf-8 -*-
"""
Contact Detection Module

This module implements spatio-temporal distance-based patient contact detection functionality, serving as the core component of the patient companion analysis system.
The module provides a unified contact detection framework supporting multiple detection modes.

Main Features:
==============
1. Hospital Companion Detection
   - Detects patient pairs appearing at the same hospital simultaneously

2. Home Companion Detection
   - Detects patient pairs with neighboring residences

3. Journey Contact Detection
   - Detects spatio-temporal contacts across complete patient trajectories

Data Flow:
==========
Input: Trajectory data in Parquet format
  ├─ Required fields: id (patient ID), lat (latitude), lon (longitude)
  └─ Optional field: datetime (timestamp)

Processing Steps:
  1. Load trajectory data
  2. Select detection mode based on configuration
  3. Build spatial index (BallTree)
  4. Find patient pairs meeting threshold conditions
  5. Deduplicate and format output

Output: Patient pair data in Parquet format
  ├─ AID, BID: IDs of patient pairs
  ├─ AID_longitude, AID_latitude: Coordinates of patient A
  ├─ BID_longitude, BID_latitude: Coordinates of patient B
  ├─ Distance: Distance between patient pairs (meters)
  └─ AID_start_time, BID_start_time: Contact timestamps (optional)

Usage Examples:
===============
>>> # Hospital contact detection
>>> config = DetectorConfig.for_hospital(
...     input_dir='data/hospital_trajectories',
...     output_dir='data/hospital_companions',
...     distance_threshold=50.0,
...     time_threshold_minutes=30
... )
>>> detector = ContactDetector(config)
>>> detector.detect_from_file('city_date.parquet')

>>> # Home contact detection
>>> config = DetectorConfig.for_home(
...     input_dir='data/residence',
...     output_dir='data/home_companions',
...     distance_threshold=100.0
... )
>>> detector = ContactDetector(config)
>>> detector.detect_from_file('city_date.parquet')
"""

import gc
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl
from tqdm import tqdm

from .geo_utils import (
    find_nearby_pairs,
    deduplicate_pairs,
    haversine_distance,
    build_ball_tree,
    to_radians,
)


COMPANION_RESULT_SCHEMA = [
    'AID', 'BID', 'AID_longitude', 'AID_latitude',
    'BID_longitude', 'BID_latitude', 'Distance',
    'AID_start_time', 'BID_start_time'
]

COMPANION_RESULT_SCHEMA_NO_TIME = [
    'AID', 'BID', 'AID_longitude', 'AID_latitude',
    'BID_longitude', 'BID_latitude', 'Distance'
]


@dataclass
class DetectorConfig:
    """
    Contact Detector Configuration Class

    Encapsulates all parameter configurations for contact detection, supporting flexible configuration for multiple detection modes.

    Basic Parameters:
        output_dir (str): Output result directory path
        distance_threshold (float): Distance threshold (meters), defines distance condition for spatio-temporal contact, default 50 meters
        time_threshold_minutes (Optional[float]): Time threshold (minutes), defines time window for spatio-temporal contact, default 30 minutes

    Data Source Parameters:
        input_dir (Optional[str]): Input data directory path
        ab_pairs_dir (Optional[str]): Patient pair data directory (journey contact detection specific)
        full_trajectory_dir (Optional[str]): Complete trajectory data directory (journey contact detection specific)
        hospital_trajectory_dir (Optional[str]): Hospital trajectory data directory (journey contact detection specific)
        home_coords_dir (Optional[str]): Residence coordinate data directory (journey contact detection specific)

    Field Mapping Parameters:
        id_column (str): Patient ID column name, default 'id'
        latitude_column (str): Latitude column name, default 'lat'
        longitude_column (str): Longitude column name, default 'lon'
        time_column (Optional[str]): Timestamp column name, default 'datetime', if None then no time filtering is used

    Optimization Parameters:
        window_length_minutes (Optional[float]): Time window length (minutes), default is 3 times time_threshold_minutes
        num_processes (int): Number of processes for parallel processing, default 24
        home_radius (float): Residence exclusion radius (meters), default 300 meters

    Class Methods (Configuration Factory):
        for_hospital(): Create hospital contact detection configuration
        for_home(): Create residence contact detection configuration
        for_journey(): Create journey contact detection configuration

    Property Methods:
        time_threshold_seconds: Convert time threshold to seconds

    Usage Examples:
        >>> # Hospital contact detection configuration
        >>> config = DetectorConfig.for_hospital(
        ...     input_dir='data/hospital',
        ...     output_dir='data/output',
        ...     distance_threshold=50.0,
        ...     time_threshold_minutes=30
        ... )
        >>> # Residence contact detection configuration
        >>> config = DetectorConfig.for_home(
        ...     input_dir='data/residence',
        ...     output_dir='data/output',
        ...     distance_threshold=100.0
        ... )
    """
    output_dir: str
    distance_threshold: float = 50.0
    time_threshold_minutes: Optional[float] = 30
    input_dir: Optional[str] = None
    id_column: str = 'id'
    latitude_column: str = 'lat'
    longitude_column: str = 'lon'
    time_column: Optional[str] = 'datetime'
    window_length_minutes: Optional[float] = None
    num_processes: int = 24
    ab_pairs_dir: Optional[str] = None
    full_trajectory_dir: Optional[str] = None
    hospital_trajectory_dir: Optional[str] = None
    home_coords_dir: Optional[str] = None
    home_radius: float = 300.0

    def __post_init__(self):
        """
        Post-initialization processing, automatically calculates time window length

        If window_length_minutes is not specified, automatically set to 3 times time_threshold_minutes
        to ensure the time window covers enough trajectory points.
        """
        if self.window_length_minutes is None and self.time_threshold_minutes:
            self.window_length_minutes = self.time_threshold_minutes * 3

    @property
    def time_threshold_seconds(self) -> Optional[float]:
        """
        Convert time threshold from minutes to seconds

        Returns:
            Optional[float]: Time threshold (seconds), returns None if time threshold is not set
        """
        return self.time_threshold_minutes * 60 if self.time_threshold_minutes else None

    @classmethod
    def for_hospital(cls, input_dir: str, output_dir: str,
                     distance_threshold: float = 50.0,
                     time_threshold_minutes: float = 30.0, **kwargs) -> 'DetectorConfig':
        """
        Create hospital contact detection configuration (factory method)

        Parameters:
            input_dir (str): Hospital trajectory data input directory
            output_dir (str): Output result directory
            distance_threshold (float): Distance threshold (meters), default 50 meters
            time_threshold_minutes (float): Time threshold (minutes), default 30 minutes
            **kwargs: Other configuration parameters

        Returns:
            DetectorConfig: Configured detector configuration object
        """
        return cls(
            output_dir=output_dir,
            distance_threshold=distance_threshold,
            time_threshold_minutes=time_threshold_minutes,
            input_dir=input_dir,
            **kwargs
        )

    @classmethod
    def for_home(cls, input_dir: str, output_dir: str, distance_threshold: float = 50.0,
                 lat_column: str = 'lat', lon_column: str = 'lon', **kwargs) -> 'DetectorConfig':
        """
        Create residence contact detection configuration (factory method)

        Parameters:
            input_dir (str): Residence coordinate data input directory
            output_dir (str): Output result directory
            distance_threshold (float): Distance threshold (meters), default 50 meters
            lat_column (str): Latitude column name, default 'lat'
            lon_column (str): Longitude column name, default 'lon'
            **kwargs: Other configuration parameters

        Returns:
            DetectorConfig: Configured detector configuration object

        Note:
            Residence detection does not use time filtering, therefore time_threshold_minutes and time_column are both set to None
        """
        return cls(
            output_dir=output_dir,
            distance_threshold=distance_threshold,
            time_threshold_minutes=None,
            input_dir=input_dir,
            time_column=None,
            latitude_column=lat_column,
            longitude_column=lon_column,
            **kwargs
        )

    @classmethod
    def for_journey(cls, ab_pairs_dir: str, full_trajectory_dir: str,
                    hospital_trajectory_dir: str, home_coords_dir: str, output_dir: str,
                    distance_threshold: float = 50.0,
                    time_threshold_minutes: float = 30,
                    home_radius: float = 300.0, **kwargs) -> 'DetectorConfig':
        """
        Create journey contact detection configuration (factory method)

        Used to detect spatio-temporal contacts between patient pairs outside of hospital and residence areas.

        Parameters:
            ab_pairs_dir (str): Patient pair data directory
            full_trajectory_dir (str): Complete trajectory data directory
            hospital_trajectory_dir (str): Hospital trajectory data directory, used to exclude contacts within hospital range
            home_coords_dir (str): Residence coordinate data directory, used to exclude contacts within residence range
            output_dir (str): Output result directory
            distance_threshold (float): Distance threshold (meters), default 50 meters
            time_threshold_minutes (float): Time threshold (minutes), default 30 minutes
            home_radius (float): Residence exclusion radius (meters), default 300 meters
            **kwargs: Other configuration parameters

        Returns:
            DetectorConfig: Configured detector configuration object
        """
        return cls(
            output_dir=output_dir,
            distance_threshold=distance_threshold,
            time_threshold_minutes=time_threshold_minutes,
            ab_pairs_dir=ab_pairs_dir,
            full_trajectory_dir=full_trajectory_dir,
            hospital_trajectory_dir=hospital_trajectory_dir,
            home_coords_dir=home_coords_dir,
            home_radius=home_radius,
            **kwargs
        )


@dataclass
class ContactRecord:
    """
    Contact Record Data Class

    Encapsulates single contact record information for patient pairs, including detailed spatio-temporal positions and contact types.

    Attributes:
        aid (str): Patient A's ID
        bid (str): Patient B's ID
        aid_lon (float): Patient A's longitude
        aid_lat (float): Patient A's latitude
        bid_lon (float): Patient B's longitude
        bid_lat (float): Patient B's latitude
        aid_time (str): Patient A's timestamp
        bid_time (str): Patient B's timestamp
        mid_lat (float): Contact point center latitude
        mid_lon (float): Contact point center longitude
        mid_time (str): Contact time midpoint
        distance (float): Distance between patients (meters)
        time_diff_minutes (float): Time difference (minutes)
        contact_type (str): Contact type, default "journey"

    Methods:
        to_dict(): Convert contact record to dictionary format
    """
    aid: str
    bid: str
    aid_lon: float
    aid_lat: float
    bid_lon: float
    bid_lat: float
    aid_time: str
    bid_time: str
    mid_lat: float
    mid_lon: float
    mid_time: str
    distance: float
    time_diff_minutes: float
    contact_type: str = "journey"

    def to_dict(self) -> Dict:
        """
        Convert contact record to dictionary format

        Returns:
            Dict: Dictionary containing all contact information with keys matching standard output format
        """
        return {
            "AID": self.aid, "BID": self.bid,
            "AID_longitude": self.aid_lon, "AID_latitude": self.aid_lat,
            "BID_longitude": self.bid_lon, "BID_latitude": self.bid_lat,
            "AID_start_time": self.aid_time, "BID_start_time": self.bid_time,
            "contact_mid_lat": self.mid_lat, "contact_mid_lon": self.mid_lon,
            "contact_mid_time": self.mid_time,
            "distance": self.distance, "time_diff_minutes": self.time_diff_minutes,
            "contact_type": self.contact_type,
        }


@dataclass
class DetectionResult:
    """Detection result"""
    city: str
    date: str
    status: str
    processed_pairs: int = 0
    contact_count: int = 0
    processing_time: float = 0.0
    error: str = ""


def create_time_windows(df: pl.DataFrame, time_threshold: float,
                        window_length: float, time_column: str) -> List[Dict]:
    """Create time windows"""
    if len(df) == 0:
        return []

    window_delta = timedelta(minutes=window_length)
    step_delta = window_delta - timedelta(minutes=time_threshold)
    time_series = df[time_column]
    min_time, max_time = time_series.min(), time_series.max()

    windows = []
    current = min_time
    while current <= max_time:
        end = current + window_delta
        if ((time_series >= current) & (time_series <= end)).sum() > 1:
            windows.append({'start_time': current, 'end_time': end})
        current += step_delta
    return windows


def process_window_worker(args):
    """Multi-process worker for processing single time window"""
    window_indices, data_dict, config = args
    if len(window_indices) <= 1:
        return []

    return find_nearby_pairs(
        data_dict['ids'][window_indices],
        data_dict['lats'][window_indices],
        data_dict['lons'][window_indices],
        data_dict['times'][window_indices],
        config.distance_threshold,
        config.time_threshold_seconds,
        include_time=True
    )


class ContactDetector:
    """Unified contact detector"""

    def __init__(self, config: DetectorConfig):
        self.config = config
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    @property
    def use_time(self) -> bool:
        return self.config.time_column is not None and self.config.time_threshold_minutes is not None

    @property
    def result_schema(self) -> List[str]:
        return COMPANION_RESULT_SCHEMA if self.use_time else COMPANION_RESULT_SCHEMA_NO_TIME

    @property
    def is_journey_mode(self) -> bool:
        return self.config.ab_pairs_dir is not None

    def detect_from_file(self, file_path: str) -> Dict:
        start_time = time.time()
        file_name = Path(file_path).stem
        cfg = self.config

        # Read using lazy API
        lf = pl.scan_parquet(file_path)
        lf = lf.filter(pl.col(cfg.latitude_column).is_not_null() & pl.col(cfg.longitude_column).is_not_null())
        if self.use_time:
            lf = lf.filter(pl.col(cfg.time_column).is_not_null())

        # Collect to DataFrame
        df = lf.collect()
        if df is None or len(df) == 0:
            return self._make_file_result(file_name, 'empty', 0, time.time() - start_time)

        if self.use_time:
            pairs = self._detect_with_windows_only(df)
        else:
            pairs = self._detect_simple(df)

        pairs = deduplicate_pairs(pairs, self.result_schema)
        output_file = self._save_file_result(file_name, pairs)
        return self._make_file_result(file_name, 'success', len(pairs), time.time() - start_time, output_file)

    def _detect_simple(self, df: pl.DataFrame) -> List[Tuple]:
        cfg = self.config
        return find_nearby_pairs(
            df[cfg.id_column].to_numpy(),
            df[cfg.latitude_column].to_numpy(),
            df[cfg.longitude_column].to_numpy(),
            df[cfg.time_column].to_numpy() if self.use_time else None,
            cfg.distance_threshold,
            cfg.time_threshold_seconds,
            include_time=self.use_time
        )

    def _detect_with_windows_only(self, df: pl.DataFrame) -> List[Tuple]:
        cfg = self.config
        windows = create_time_windows(df, cfg.time_threshold_minutes, cfg.window_length_minutes, cfg.time_column)

        if cfg.num_processes > 1 and len(windows) > 1:
            return self._process_windows_multi_process(df, windows)

        all_pairs = []
        for window in windows:
            window_df = df.filter(
                (pl.col(cfg.time_column) >= window['start_time']) &
                (pl.col(cfg.time_column) <= window['end_time'])
            )
            if len(window_df) <= 1:
                continue

            pairs = find_nearby_pairs(
                window_df[cfg.id_column].to_numpy(),
                window_df[cfg.latitude_column].to_numpy(),
                window_df[cfg.longitude_column].to_numpy(),
                window_df[cfg.time_column].to_numpy(),
                cfg.distance_threshold,
                cfg.time_threshold_seconds
            )
            all_pairs.extend(pairs)
        return all_pairs

    def _process_windows_multi_process(self, group_df: pl.DataFrame, windows: List[Dict]) -> List[Tuple]:
        cfg = self.config
        data_dict = {
            'ids': group_df[cfg.id_column].to_numpy(),
            'lats': group_df[cfg.latitude_column].to_numpy(),
            'lons': group_df[cfg.longitude_column].to_numpy(),
            'times': group_df[cfg.time_column].to_numpy(),
        }

        tasks = []
        for window in windows:
            time_mask = (
                (group_df[cfg.time_column] >= window['start_time']) &
                (group_df[cfg.time_column] <= window['end_time'])
            )
            tasks.append((np.where(time_mask.to_numpy())[0], data_dict, cfg))

        with ProcessPoolExecutor(max_workers=min(cfg.num_processes, len(windows))) as executor:
            results = list(executor.map(process_window_worker, tasks))

        return [pair for pairs in results for pair in pairs]

    def _save_file_result(self, file_name: str, pairs: List[Tuple]) -> str:
        base_name = Path(file_name).stem
        output_file = os.path.join(self.config.output_dir, f"{base_name}_companion.parquet")

        if pairs:
            df = pl.DataFrame(data=pairs, orient="row", schema=self.result_schema)
        else:
            df = pl.DataFrame(schema={
                col: pl.Utf8 if col in ['AID', 'BID'] else pl.Float64
                for col in self.result_schema
            })
        df.write_parquet(output_file)
        return output_file

    def _make_file_result(self, file_name: str, status: str, pairs_count: int,
                          proc_time: float, output_file: str = "", error: str = "") -> Dict:
        return {
            'file_name': file_name, 'status': status, 'pairs_count': pairs_count,
            'processing_time': proc_time, 'output_file': output_file, 'error': error
        }

    def _detect_files(self, progress: bool = True) -> List[Dict]:
        files = [
            os.path.join(root, f)
            for root, _, filenames in os.walk(self.config.input_dir)
            for f in filenames if f.endswith('.parquet')
        ]
        if not files:
            return []

        results = []
        iterator = tqdm(files, desc="Detecting contacts") if progress else files
        for file_path in iterator:
            results.append(self.detect_from_file(file_path))
            gc.collect()
        return results

    def load_ab_pairs(self, city: str, date: str) -> Set[Tuple[str, str]]:
        file_path = Path(self.config.ab_pairs_dir) / f"{city}_{date}_matched.parquet"
        if not file_path.exists():
            return set()

        pairs = (
            pl.scan_parquet(file_path)
            .with_columns([
                pl.when(pl.col("AID") <= pl.col("BID")).then(pl.col("AID")).otherwise(pl.col("BID")).alias("id1"),
                pl.when(pl.col("AID") <= pl.col("BID")).then(pl.col("BID")).otherwise(pl.col("AID")).alias("id2"),
            ])
            .select(["id1", "id2"])
            .unique()
            .collect()
        )
        return {(row[0], row[1]) for row in pairs.iter_rows()}

    def load_home_coordinates(self, city: str, target_ids: Set[str]) -> Dict[str, Tuple[float, float]]:
        file_path = Path(self.config.home_coords_dir) / city / f"{city}_residence.parquet"
        if not file_path.exists():
            return {}

        # Read using lazy API and filter
        lf = pl.scan_parquet(file_path)
        # Get first column as id column
        id_col = lf.collect_schema().names()[0]
        df = (
            lf.filter(pl.col(id_col).is_in(target_ids))
            .select([id_col, 'lat', 'lon'])
            .collect()
        )
        # Ensure correct coordinate order (lat, lon) based on column names rather than positions
        return {row[0]: (float(row[1]), float(row[2])) for row in df.iter_rows()}

    def load_hospital_trajectory(self, city: str, date: str, target_ids: Set[str]) -> Optional[pl.DataFrame]:
        file_path = Path(self.config.hospital_trajectory_dir) / city / f"{city}_{date}_hospital.parquet"
        if not file_path.exists():
            return None
        return (
            pl.scan_parquet(file_path)
            .filter(pl.col("id").is_in(target_ids))
            .collect()
        )

    def exclude_hospital_points(self, df_full: pl.DataFrame, df_hospital: Optional[pl.DataFrame]) -> pl.DataFrame:
        """Exclude hospital points from trajectories (exclude only when both ID and time match)"""
        if df_hospital is None:
            return df_full

        hospital_points = {(str(row[0]), str(row[3])[:19]) for row in df_hospital.iter_rows()}
        if not hospital_points:
            return df_full

        df_with_time = df_full.with_columns(pl.col("datetime").dt.strftime("%Y-%m-%d %H:%M:%S").alias("_time_str"))

        hospital_keys = {f"{pid}|{t}" for pid, t in hospital_points}
        df_with_key = df_with_time.with_columns((pl.col("id") + "|" + pl.col("_time_str")).alias("_key"))
        return df_with_key.filter(~pl.col("_key").is_in(hospital_keys)).drop(["_time_str", "_key"])

    def find_contacts_for_pair(
        self, aid: str, bid: str,
        lats: np.ndarray, lons: np.ndarray, times: np.ndarray,
        tree, id_to_indices: Dict[str, List[int]],
        home_coords: Dict[str, Tuple[float, float]]
    ) -> List[ContactRecord]:
        if aid not in id_to_indices or bid not in id_to_indices:
            return []
        if aid not in home_coords or bid not in home_coords:
            return []

        home_mid_lat = (home_coords[aid][0] + home_coords[bid][0]) / 2
        home_mid_lon = (home_coords[aid][1] + home_coords[bid][1]) / 2

        aid_indices = id_to_indices[aid]
        bid_indices_set = set(id_to_indices[bid])
        contacts = []
        radius_rad = to_radians(self.config.distance_threshold)

        for i in aid_indices:
            nearby = tree.query_radius(np.radians([[lats[i], lons[i]]]), r=radius_rad, return_distance=False)[0]

            for j in nearby:
                if j not in bid_indices_set:
                    continue

                time_diff_sec = abs((times[j] - times[i]).astype("timedelta64[s]").astype(int))
                if time_diff_sec > self.config.time_threshold_seconds:
                    continue

                dist = haversine_distance(lats[i], lons[i], lats[j], lons[j])
                if dist > self.config.distance_threshold:
                    continue

                mid_lat, mid_lon = (lats[i] + lats[j]) / 2, (lons[i] + lons[j]) / 2
                if haversine_distance(home_mid_lat, home_mid_lon, mid_lat, mid_lon) <= self.config.home_radius:
                    continue

                mid_time = times[i] + np.timedelta64(int(time_diff_sec / 2), "s")

                if aid <= bid:
                    final_aid, final_bid = aid, bid
                    aid_data, bid_data = (lons[i], lats[i], times[i]), (lons[j], lats[j], times[j])
                else:
                    final_aid, final_bid = bid, aid
                    aid_data, bid_data = (lons[j], lats[j], times[j]), (lons[i], lats[i], times[i])

                contacts.append(ContactRecord(
                    aid=final_aid, bid=final_bid,
                    aid_lon=aid_data[0], aid_lat=aid_data[1],
                    bid_lon=bid_data[0], bid_lat=bid_data[1],
                    aid_time=str(aid_data[2]), bid_time=str(bid_data[2]),
                    mid_lat=mid_lat, mid_lon=mid_lon, mid_time=str(mid_time),
                    distance=dist, time_diff_minutes=time_diff_sec / 60.0,
                ))
        return contacts

    def detect_city_date(self, city: str, date: str) -> DetectionResult:
        start_time = time.time()

        ab_pairs = self.load_ab_pairs(city, date)
        if not ab_pairs:
            return DetectionResult(city, date, "skipped", error="no_ab_pairs")

        target_ids = {pid for pair in ab_pairs for pid in pair}
        home_coords = self.load_home_coordinates(city, target_ids)

        traj_path = Path(self.config.full_trajectory_dir) / city / f"{city}_{date}_matched_trajectories.parquet"
        # Read using lazy API
        df_full = pl.scan_parquet(str(traj_path)).collect()

        df_hospital = self.load_hospital_trajectory(city, date, target_ids)
        df_full = self.exclude_hospital_points(df_full, df_hospital)

        ids = df_full["id"].to_numpy()
        lats = df_full["lat"].to_numpy()
        lons = df_full["lon"].to_numpy()
        times = df_full["datetime"].to_numpy()

        tree = build_ball_tree(lats, lons)

        id_to_indices: Dict[str, List[int]] = {}
        for i, pid in enumerate(ids):
            id_to_indices.setdefault(pid, []).append(i)

        all_contacts = []
        for aid, bid in ab_pairs:
            contacts = self.find_contacts_for_pair(aid, bid, lats, lons, times, tree, id_to_indices, home_coords)
            all_contacts.extend([c.to_dict() for c in contacts])

        if all_contacts:
            df_result = pl.DataFrame(all_contacts).unique(keep="first").sort(["AID", "BID", "contact_mid_time"])
        else:
            df_result = pl.DataFrame(schema={
                "AID": pl.Utf8, "BID": pl.Utf8,
                "AID_longitude": pl.Float64, "AID_latitude": pl.Float64,
                "BID_longitude": pl.Float64, "BID_latitude": pl.Float64,
                "AID_start_time": pl.Utf8, "BID_start_time": pl.Utf8,
                "contact_mid_lat": pl.Float64, "contact_mid_lon": pl.Float64,
                "contact_mid_time": pl.Utf8,
                "distance": pl.Float64, "time_diff_minutes": pl.Float64,
                "contact_type": pl.Utf8,
            })

        output_file = Path(self.config.output_dir) / f"{city}_{date}_final_companion.parquet"
        df_result.write_parquet(str(output_file))

        del df_full, tree
        gc.collect()

        return DetectionResult(
            city=city, date=date, status="success",
            processed_pairs=len(ab_pairs), contact_count=len(df_result),
            processing_time=time.time() - start_time,
        )

    def get_available_city_dates(self) -> List[Tuple[str, str]]:
        """Get available city-date list (filename format: {city}_{date}_matched.parquet)"""
        ab_dir = Path(self.config.ab_pairs_dir)
        pattern = re.compile(r"^([^_]+)_(\d{4}-\d{2}-\d{2})_matched$")

        city_dates = []
        for file in ab_dir.iterdir():
            if file.suffix == ".parquet":
                match = pattern.match(file.stem)
                if match:
                    city_dates.append((match.group(1), match.group(2)))
        return sorted(city_dates)

    def _detect_journey(self, progress: bool = True) -> List[DetectionResult]:
        city_dates = self.get_available_city_dates()
        if not city_dates:
            return []

        results = []
        iterator = tqdm(city_dates, desc="Contact detection") if progress else city_dates
        for city, date in iterator:
            results.append(self.detect_city_date(city, date))
        return results

    def detect_all(self, progress: bool = True):
        """Unified detection entry point"""
        if self.is_journey_mode:
            return self._detect_journey(progress)
        return self._detect_files(progress)


def detect_hospital_companions(input_dir: str, output_dir: str,
                               distance_threshold: float = 50.0,
                               time_threshold_minutes: float = 30.0,
                               num_processes: int = 24, **kwargs) -> List[Dict]:
    """Detect hospital contacts"""
    config = DetectorConfig(
        output_dir=output_dir,
        distance_threshold=distance_threshold,
        time_threshold_minutes=time_threshold_minutes,
        input_dir=input_dir,
        num_processes=num_processes,
        **kwargs
    )
    return ContactDetector(config).detect_all()


def detect_home_companions(input_dir: str, output_dir: str, distance_threshold: float = 50.0,
                           lat_column: str = 'lat', lon_column: str = 'lon',
                           num_processes: int = 24, **kwargs) -> List[Dict]:
    """Detect residence contacts"""
    config = DetectorConfig.for_home(input_dir, output_dir, distance_threshold, lat_column, lon_column, **kwargs)
    config.num_processes = num_processes
    return ContactDetector(config).detect_all()


def detect_journey_contacts(ab_pairs_dir: str, full_trajectory_dir: str,
                            hospital_trajectory_dir: str, home_coords_dir: str,
                            output_dir: str, home_radius: float = 300.0,
                            progress: bool = True) -> List[DetectionResult]:
    """Detect journey contacts"""
    config = DetectorConfig.for_journey(
        ab_pairs_dir, full_trajectory_dir, hospital_trajectory_dir,
        home_coords_dir, output_dir, home_radius=home_radius
    )
    return ContactDetector(config).detect_all(progress)