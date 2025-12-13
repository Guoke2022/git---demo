# -*- coding: utf-8 -*-
"""
Record Extraction Module

This module implements functionality to extract complete records of specific patient groups from raw trajectory data, providing a data foundation for subsequent full-course contact analysis.
Supports parallel processing and large-scale data extraction.

Main Features:
==============
1. ID Extraction
   - Extract all unique patient IDs from matched patient pair files
   - Support automatic merging and deduplication of AID and BID columns
   - Generate target patient ID lists

2. Full Record Extraction
   - Extract complete records from raw trajectory data based on patient ID lists
   - Preserve all spatiotemporal trajectory points for patients
   - Used for full-course contact analysis

3. Parallel Batch Processing
   - Support multi-process parallel extraction
   - Automatically scan directories and generate extraction tasks
   - Display progress bar to monitor processing status

Data Flow:
==========
Input:
  ├─ Matched patient pair data (Parquet): data/matched_pairs/{city}_{date}_matched.parquet
  │   Fields: AID, BID, ...
  │   Purpose: Provide target patient ID lists
  └─ Raw trajectory data (Parquet): data/origin_trajectories/{city}_{date}.parquet
      Fields: id, lat, lon, datetime, ...
      Purpose: Provide complete trajectory records

Processing Workflow:
  1. Read matched patient pair files
  2. Extract AID and BID columns, merge and deduplicate to get unique ID lists
  3. Read corresponding city-date raw trajectory data
  4. Filter: Keep records where ID is in the target list
  5. Save extraction results to output directory

Output:
  {city}_{date}_matched_trajectories.parquet
  Fields: id, lat, lon, datetime, ... (same as raw trajectory data)

Usage Examples:
===============
>>> # Basic usage
>>> from core.record_extractor import extract_companion_records
>>> results = extract_companion_records(
...     trajectory_dir='data/origin_trajectories',
...     accompany_dir='data/matched_pairs',
...     output_dir='data/extracted_trajectories',
...     workers=8
... )
>>> for result in results:
...     if result.success:
...         print(f"{result.city} {result.date}: Extracted {result.record_count} records")

"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import polars as pl
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


@dataclass
class ExtractConfig:
    """Extractor configuration"""
    trajectory_dir: str  # Trajectory data directory
    accompany_dir: str  # Companion patient matching results directory
    output_dir: str  # Output directory
    id_column: str = 'id'  # ID column name in trajectory data
    workers: int = 1  # Number of parallel worker processes


@dataclass
class ExtractResult:
    """Extraction result"""
    city: str
    date: str
    success: bool
    record_count: int = 0
    unique_id_count: int = 0
    original_records: int = 0
    accompany_pairs: int = 0
    output_file: str = ""
    error: str = ""


class RecordExtractor:
    """Record extractor"""

    def __init__(self, config: ExtractConfig):
        self.config = config
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def extract_unique_ids(self, accompany_file: Path) -> pl.Series:
        """Extract unique IDs from companion matching file"""
        df = pl.read_parquet(accompany_file)
        return (
            df.select(['AID', 'BID'])
            .unpivot()
            .get_column('value')
            .unique()
        )

    def extract_all_records(
        self,
        city: str,
        date: str,
        trajectory_file: Path,
        accompany_file: Path
    ) -> ExtractResult:
        """Extract all records of companion patients"""
        try:
            unique_ids = self.extract_unique_ids(accompany_file)
            traj_df = pl.read_parquet(trajectory_file)
            matched_records = traj_df.filter(pl.col(self.config.id_column).is_in(unique_ids))

            output_dir = Path(self.config.output_dir) / city
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{city}_{date}_matched_trajectories.parquet"
            matched_records.write_parquet(output_file)

            return ExtractResult(
                city=city,
                date=date,
                success=True,
                record_count=len(matched_records),
                unique_id_count=len(unique_ids),
                original_records=len(traj_df),
                accompany_pairs=pl.read_parquet(accompany_file).height,
                output_file=str(output_file)
            )

        except Exception as e:
            return ExtractResult(
                city=city,
                date=date,
                success=False,
                error=str(e)
            )


def _process_single_city_date(args: Tuple) -> ExtractResult:
    """Process single city-date pair"""
    city, date, traj_file, accompany_file, output_dir, id_column = args

    config = ExtractConfig(
        trajectory_dir="",
        accompany_dir="",
        output_dir=output_dir,
        id_column=id_column
    )
    extractor = RecordExtractor(config)
    return extractor.extract_all_records(city, date, traj_file, accompany_file)


# Convenience function
def extract_companion_records(
    trajectory_dir: str,
    accompany_dir: str,
    output_dir: str,
    id_column: str = 'id',
    workers: int = 1,
    progress: bool = True
) -> List[ExtractResult]:
    """Batch extract companion patient records"""
    traj_path = Path(trajectory_dir)
    accompany_path = Path(accompany_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    tasks = []
    for city_dir in traj_path.iterdir():
        if not city_dir.is_dir():
            continue

        city_name = city_dir.name
        for traj_file in city_dir.glob("*.parquet"):
            date_str = traj_file.stem.replace(f"{city_name}_", "")
            accompany_file = accompany_path / f"{city_name}_{date_str}_matched.parquet"
            if accompany_file.exists():
                tasks.append((
                    city_name, date_str,
                    traj_file, accompany_file,
                    output_dir, id_column
                ))

    if not tasks:
        return []

    results = []
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_single_city_date, task): task for task in tasks}
            if progress:
                for i in range(len(tasks)):
                    results.append(futures.popitem()[1].result())
            else:
                for future in as_completed(futures):
                    results.append(future.result())
    else:
        iterator = tqdm(tasks, desc="Extracting records") if progress else tasks
        for task in iterator:
            results.append(_process_single_city_date(task))

    return results


