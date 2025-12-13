# -*- coding: utf-8 -*-
"""
Companion Identification Core Module Package

This package provides all core functional modules of the companion identification,
implementing a complete workflow from spatio-temporal contact detection to patient role identification.

Module Architecture:
==================
1. geo_utils
   - Provides basic geographic distance calculation and spatial indexing functionality
   - Based on Haversine formula and BallTree spatial indexing
   - Supports efficient nearest neighbor queries and spatio-temporal filtering

2. contact_detector
   - Implements multi-mode contact detection (hospital, residence, entire journey)
   - Performs spatio-temporal contact analysis based on geo_utils spatial indexing
   - Supports multi-process parallel processing and time window optimization

3. pair_matcher
   - Executes cross-matching of patient pairs from multiple data sources
   - Identifies patient pairs that simultaneously meet multiple conditions
   - Supports automatic file scanning and batch processing

4. record_extractor
   - Extracts target patient records from raw trajectory data

5. patient_classifier
   - Analyzes patient contact networks based on network graph theory
   - Automatically identifies network structure types and patient roles
   - Calculates companion ratios and output classification results

Data Flow:
=========
Raw Trajectory Data
    ↓
[contact_detector] Spatio-temporal Contact Detection
    ↓
Patient Pair Data
    ↓
[pair_matcher] Multi-condition Matching
    ↓
Matched Patient Pairs
    ↓
[record_extractor] Trajectory Extraction
    ↓
Complete Trajectory Data
    ↓
[contact_detector] Entire Journey Contact Detection
    ↓
Final Patient Pairs
    ↓
[patient_classifier] Network Analysis and Role Classification
    ↓
Patient/Companion Patient Classification Results

Export Interface:
===============
Utility Functions:
  - haversine_distance: Calculate spherical distance between two points
  - to_radians: Convert meters to radians
  - build_ball_tree: Build spatial index
  - find_nearby_points: Range queries

Configuration Classes:
  - DetectorConfig: Contact detector configuration

Core Classes:
  - ContactDetector: Contact detector
  - PairMatcher: Patient pair matcher
  - RecordExtractor: Record extractor
  - RatioCalculator: Ratio calculator

Usage Examples:
==============
>>> # Import core modules
>>> from core import ContactDetector, DetectorConfig
>>>
>>> # Configure and run contact detection
>>> config = DetectorConfig.for_hospital(
...     input_dir='data/hospital_trajectories',
...     output_dir='data/hospital_companions',
...     distance_threshold=50,
...     time_threshold_minutes=30
... )
>>> detector = ContactDetector(config)
>>> results = detector.detect_from_file('Beijing_2024-01-01.csv')
>>>
>>> # Or use complete pipeline (see src/3.family_accompany_identification.py)
>>> from ..3.family_accompany_identification import CompanionAnalysisPipeline, PipelineConfig
>>> pipeline = CompanionAnalysisPipeline(PipelineConfig())
>>> pipeline.run_full_pipeline()

Dependencies:
=============
- geo_utils: No internal dependencies (base layer)
- contact_detector: Depends on geo_utils
- pair_matcher: No internal dependencies (independent module)
- record_extractor: No internal dependencies (independent module)
- patient_classifier: No internal dependencies (independent module)

External Dependencies:
====================
- numpy: Numerical computation
- polars: High-performance data processing
- scikit-learn: BallTree spatial indexing
- networkx: Network graph theory analysis
- haversine: Haversine distance calculation

Version: v1.0.0
"""

from .geo_utils import (
    haversine_distance,
    to_radians,
    build_ball_tree,
    find_nearby_points,
)
from .contact_detector import DetectorConfig, ContactDetector
from .pair_matcher import PairMatcher
from .record_extractor import RecordExtractor
from .patient_classifier import RatioCalculator

__all__ = [
    # 地理工具
    'haversine_distance',
    'to_radians',
    'build_ball_tree',
    'find_nearby_points',
    # 配置类
    'DetectorConfig',
    # 核心类
    'ContactDetector',
    'PairMatcher',
    'RecordExtractor',
    'RatioCalculator',
]
