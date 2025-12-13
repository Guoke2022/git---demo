# -*- coding: utf-8 -*-
"""
ABID Pair Matching Module

This module implements cross-matching functionality for patient pairs, used to identify patient pairs that satisfy both hospital contact and residence contact conditions.

Main Features:
==============
1. Dual-Source Matching
   - Matches patient pairs from two different data sources
   - Typically used to match hospital companions and residence companions
   - Identifies patient pairs that satisfy both conditions simultaneously

2. Intelligent Filename Parsing
   - Automatically parses city and date information from filenames
   - Supports multiple naming formats (residence format, hospital format)
   - Establishes city-date to file mapping relationships

3. Batch Matching Processing
   - Automatically scans all files in two directories
   - Aligns files by city-date for matching
   - Generates matching tasks and executes them in batches

4. Matching Strategy
   - Direct matching: (AID1, BID1) appears in both data sources
   - Cross matching: Supports symmetric matching of (AID, BID) and (BID, AID)
   - Ensures uniqueness: Deduplication and standardized output

Data Flow:
========
Input:
  ├─ Residence companion data (CSV): Contains patient pairs in residential proximity
  │   Format: CityName_residence_companion.csv
  │   Fields: AID, BID, AID_longitude, AID_latitude, ...
  └─ Hospital companion data (CSV): Contains patient pairs with hospital contact
      Format: CityName_Date_hospital_companion.csv
      Fields: AID, BID, AID_longitude, AID_latitude, ...

Processing Flow:
  1. Scan two directories, parse filenames to extract city and date
  2. Build city-date to file path mapping table
  3. For each city-date combination, find corresponding two files
  4. Read patient pair data from the two files
  5. Execute ABID pair matching (set intersection operation)
  6. Save matching results to output directory

Output:
  CityName_Date_residence_hospital_matched.csv
  Fields: AID, BID, AID_longitude, AID_latitude, BID_longitude, BID_latitude, Distance

Usage Examples:
==========
>>> # Basic usage
>>> from core.pair_matcher import match_home_hospital_pairs
>>> results = match_home_hospital_pairs(
...     home_folder='data/residence_companions',
...     hospital_folder='data/hospital_companions',
...     output_folder='data/matched_pairs'
... )
>>> for result in results:
...     print(f"{result.city_name} {result.date}: matched {result.matched_count} pairs")

>>> # Single city-date matching
>>> from core.pair_matcher import PairMatcher
>>> matcher = PairMatcher(config)
>>> result = matcher.process_single('Beijing', '2024-01-01')

"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl


@dataclass
class MatcherConfig:
    """Matcher Configuration"""
    home_folder: str
    hospital_folder: str
    output_folder: str

    @property
    def home_path(self) -> Path:
        return Path(self.home_folder)

    @property
    def hospital_path(self) -> Path:
        return Path(self.hospital_folder)

    @property
    def output_path(self) -> Path:
        path = Path(self.output_folder)
        path.mkdir(parents=True, exist_ok=True)
        return path


@dataclass
class MatchResult:
    """Matching Result"""
    city_name: str
    date: str
    matched_pairs: pl.DataFrame
    file1_path: str
    file2_path: str
    total_rows_file1: int
    total_rows_file2: int
    matched_count: int


class FilenameParser:
    """Filename Parser"""
    HOME_PATTERN = re.compile(r"^([^_]+)_residence_companion$")
    HOSPITAL_PATTERN = re.compile(r"^([^_]+)_(\d{4}-\d{2}-\d{2})_hospital_companion$")

    @classmethod
    def parse_home_file(cls, filename: str) -> Optional[str]:
        """Extract city name from home filename"""
        match = cls.HOME_PATTERN.match(filename)
        return match.group(1) if match else None

    @classmethod
    def parse_hospital_file(cls, filename: str) -> Optional[Tuple[str, str]]:
        """Extract city name and date from hospital filename"""
        match = cls.HOSPITAL_PATTERN.match(filename)
        return (match.group(1), match.group(2)) if match else None


class FileScanner:
    """File Scanner"""

    @staticmethod
    def scan_home_files(folder: Path) -> Dict[str, Path]:
        """Scan home files, return {city: file path}"""
        if not folder.exists():
            return {}

        files = {}
        for path in folder.glob("*.parquet"):
            city = FilenameParser.parse_home_file(path.stem)
            if city:
                files[city] = path

        return files

    @staticmethod
    def scan_hospital_files(folder: Path) -> Dict[str, Dict[str, Path]]:
        """Scan hospital files, return {city: {date: file path}}"""
        if not folder.exists():
            return {}

        files: Dict[str, Dict[str, Path]] = {}
        for path in folder.glob("*.parquet"):
            parsed = FilenameParser.parse_hospital_file(path.stem)
            if parsed:
                city, date = parsed
                files.setdefault(city, {})[date] = path

        return files


class PairMatcher:
    """ABID Pair Matcher"""

    REQUIRED_COLUMNS = {"AID", "BID"}

    def __init__(self, config: MatcherConfig):
        self.config = config

    @classmethod
    def match_pairs(cls, df1: pl.DataFrame, df2: pl.DataFrame) -> pl.DataFrame:
        """Match ABID pairs in two DataFrames (direct matching + cross matching)"""
        pairs1 = df1.select(["AID", "BID"]).unique()
        pairs2 = df2.select(["AID", "BID"]).unique()

        direct = pairs1.join(pairs2, on=["AID", "BID"], how="inner")
        cross = pairs1.join(
            pairs2,
            left_on=["AID", "BID"],
            right_on=["BID", "AID"],
            how="inner"
        )

        return pl.concat([direct, cross]).unique()

    @classmethod
    def load_and_validate(cls, path: Path) -> Optional[pl.DataFrame]:
        """Load and validate Parquet file"""
        try:
            df = pl.read_parquet(path)
            if not cls.REQUIRED_COLUMNS.issubset(df.columns):
                return None
            return df
        except Exception:
            return None

    def build_tasks(self) -> List[Tuple[str, str, Path, Path]]:
        """Build matching task list"""
        home_files = FileScanner.scan_home_files(self.config.home_path)
        hospital_files = FileScanner.scan_hospital_files(self.config.hospital_path)

        common_cities = set(home_files) & set(hospital_files)

        tasks = []
        for city in common_cities:
            home_file = home_files[city]
            for date, hospital_file in hospital_files[city].items():
                tasks.append((city, date, home_file, hospital_file))

        return tasks

    def process_single(self, city: str, date: str, home_file: Path, hospital_file: Path) -> Optional[MatchResult]:
        """Process single matching task"""
        df_home = self.load_and_validate(home_file)
        df_hospital = self.load_and_validate(hospital_file)

        if df_home is None or df_hospital is None:
            return None

        matched = self.match_pairs(df_home, df_hospital)

        result = MatchResult(
            city_name=city,
            date=date,
            matched_pairs=matched,
            file1_path=str(home_file),
            file2_path=str(hospital_file),
            total_rows_file1=len(df_home),
            total_rows_file2=len(df_hospital),
            matched_count=len(matched),
        )

        # Save results
        output_file = self.config.output_path / f"{city}_{date}_matched.parquet"
        result.matched_pairs.write_parquet(output_file)

        return result

    def run(self, progress: bool = True) -> List[MatchResult]:
        """Run matching process"""
        from tqdm import tqdm

        tasks = self.build_tasks()
        if not tasks:
            return []

        results = []
        iterator = tqdm(tasks, desc="Matching AB pairs") if progress else tasks

        for city, date, home_file, hospital_file in iterator:
            result = self.process_single(city, date, home_file, hospital_file)
            if result:
                results.append(result)

        return results


# Convenience function
def match_home_hospital_pairs(home_folder: str,  hospital_folder: str, output_folder: str, progress: bool = True) -> List[MatchResult]:
    """
    Match home and hospital ABID pairs

    Parameters:
        home_folder: Home companion files directory
        hospital_folder: Hospital companion files directory
        output_folder: Output directory
        progress: Whether to show progress bar

    Returns:
        List of matching results
    """
    config = MatcherConfig(
        home_folder=home_folder,
        hospital_folder=hospital_folder,
        output_folder=output_folder
    )
    matcher = PairMatcher(config)
    return matcher.run(progress)
