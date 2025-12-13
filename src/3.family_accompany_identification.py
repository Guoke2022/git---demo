# -*- coding: utf-8 -*-
"""
Main Pipeline for Companion Patient Analysis

This module implements companion patient identification based on spatiotemporal contact analysis,
identifying patient pairs with spatiotemporal contact relationships through a multi-stage pipeline,
automatically classifying patients and companions using network graph theory methods,
and finally calculating companion ratios.

Note:
=================
This version does not include processing optimizations for ultra-large-scale data
to clearly demonstrate the processing workflow.
And all data used in this step are simulated data.

Core Pipeline Steps:
=================
1. Hospital Companion Detection
   - Identify patient pairs visiting the same hospital simultaneously based on spatiotemporal overlap
   - Use BallTree spatial index to accelerate neighbor queries

2. Home Companion Detection
   - Identify patient pairs with nearby residences based on geographic distance
   - Calculate spherical distance using Haversine formula

3. ABID Pair Matching
   - Match patient pairs that satisfy both hospital contact and residence proximity conditions
   - Direct matching and cross-matching strategies

4. Trajectory Record Extraction
   - Extract complete spatiotemporal trajectory records of matched patient pairs
   - Provide data foundation for journey contact analysis

5. Journey Contact Detection
   - Analyze patient contact situations outside hospital and non-residential areas
   - Comprehensively evaluate overall contact frequency of patient pairs

6. Companion Ratio Calculation
   - Build patient contact network graph based on NetworkX
   - Analyze network structure through graph theory metrics (degree centrality, betweenness centrality, eigenvector centrality)
   - Automatically identify network types: PAIR, STAR, DENSE, MIXED
   - Classify patient and companion roles based on network structure
   - Calculate companion patient ratios for each city-date combination

Data Flow:
==========
Raw Trajectory Data
    ↓
[Step 1] Hospital Companion Detection → data/companion_pairs/1_hospital/{rule}/
    ↓
[Step 2] Home Companion Detection → data/companion_pairs/2_residence/{rule}/
    ↓
[Step 3] ABID Pair Matching → data/companion_pairs/3_residence_hospital_matched/{rule}/
    ↓
[Step 4] Trajectory Record Extraction → data/trajectories/residence_hospital_matched_all_trajectories/{rule}/
    ↓
[Step 5] Journey Contact Detection → data/companion_pairs/4_final_pairs/{rule}/
    ↓
[Step 6] Companion Ratio Calculation → data/companions/{rule}/
    ↓
Final Output: Patient/Companion Classification Results + Companion Ratio Statistics

"""

import logging
import multiprocessing as mp
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from accompany_identification_core.contact_detector import (
    detect_hospital_companions,
    detect_home_companions,
    detect_journey_contacts,
)
from accompany_identification_core.pair_matcher import match_home_hospital_pairs
from accompany_identification_core.record_extractor import extract_companion_records
from accompany_identification_core.patient_classifier import calculate_companion_ratio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """
    Pipeline Configuration Parameter Class

    This class encapsulates all configuration parameters for the companion patient analysis pipeline,
    supporting multi-threshold combination analysis.

    Attributes:
        base_data_dir (str): Root data directory path, default is "data"
        distance_thresholds (List[int]): List of distance thresholds (in meters) for defining spatiotemporal contact distance conditions
        time_thresholds (List[int]): List of time thresholds (in minutes) for defining spatiotemporal contact time windows
        num_processes (int): Number of parallel processes, default 24
        home_radius (float): Residence radius (in meters) for excluding contacts within residential areas, default 300 meters

    Methods:
        rule_combinations: Generate list of rule names for all distance-time threshold combinations

    Usage Example:
        >>> config = PipelineConfig(
        ...     distance_thresholds=[50, 100],
        ...     time_thresholds=[30, 60],
        ...     num_processes=24
        ... )
        >>> config.rule_combinations
        ['50m30m', '50m60m', '100m30m', '100m60m']
    """
    base_data_dir: str = "data"
    distance_thresholds: List[int] = None
    time_thresholds: List[int] = None
    num_processes: int = 24
    home_radius: float = 300.0

    def __post_init__(self):
        """
        Post-initialization processing, setting default thresholds

        If threshold lists are not specified, default values are used:
        - Distance threshold: 50 meters
        - Time threshold: 30 minutes
        """
        if self.distance_thresholds is None:
            self.distance_thresholds = [50]
        if self.time_thresholds is None:
            self.time_thresholds = [30]

    @property
    def rule_combinations(self) -> List[str]:
        """
        Generate rule names for all distance-time threshold combinations

        Rule naming format: {distance}m{time}m
        For example: "50m30m" represents a contact rule with 50 meters distance and 30 minutes time

        Returns:
            List[str]: List of rule names
        """
        return [
            f"{d}m{t}m"
            for d in self.distance_thresholds
            for t in self.time_thresholds
        ]


class CompanionAnalysisPipeline:
    """
    Main Class for Companion Patient Analysis Pipeline

    This class implements the complete six-step pipeline processing flow, coordinating various core modules
    to complete all analysis tasks from raw trajectory data to final patient role classification.

    Attributes:
        config (PipelineConfig): Pipeline configuration object
        base_dir (Path): Root data directory path object

    Main Methods:
        step1_detect_hospital_companions(): Hospital companion detection
        step2_detect_home_companions(): Residence companion detection
        step3_match_pairs(): ABID pair matching
        step4_extract_records(): Trajectory record extraction
        step5_detect_contacts(): Journey contact detection
        step6_calculate_ratio(): Companion ratio calculation
        run_full_pipeline(): Execute complete pipeline

    Workflow:
        1. Receive configuration parameters
        2. Execute six processing steps sequentially
        3. Clean up intermediate files at key nodes (optional)
        4. Output final analysis results

    Usage Example:
        >>> config = PipelineConfig(distance_thresholds=[50], time_thresholds=[30])
        >>> pipeline = CompanionAnalysisPipeline(config)
        >>> results = pipeline.run_full_pipeline(cleanup_intermediate=True)
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline object

        Parameters:
            config (PipelineConfig): Pipeline configuration object containing all necessary parameter settings
        """
        self.config = config
        self.base_dir = Path(config.base_data_dir)

    def step1_detect_hospital_companions(
        self,
        input_dir: Optional[str] = None,
        output_base_dir: Optional[str] = None,
    ) -> dict:
        """
        Step 1: Detect Hospital Companions

        Identify patient pairs appearing at the same hospital during the same time period based on spatiotemporal overlap analysis.
        This step uses BallTree spatial index and time window grouping strategies to efficiently process large-scale trajectory data.

        Parameters:
            input_dir (Optional[str]): Hospital trajectory data input directory, default is data/trajectories/hospital_trajectories
            output_base_dir (Optional[str]): Output result base directory, default is data/companion_pairs/1_hospital

        Returns:
            dict: Processing result statistics, containing the following fields:
                - step (str): Step name
                - total_files (int): Total number of processed files
                - success_files (int): Number of successfully processed files

        Processing Logic:
            1. Iterate through all distance and time threshold combinations
            2. Call contact detector for each combination
            3. Output formatted patient pair list
        """
        logger.info("Step 1: Detect hospital companions")

        input_dir = input_dir or str(self.base_dir / "trajectories/hospital_trajectories")
        output_base_dir = output_base_dir or str(self.base_dir / "companion_pairs/1_hospital")

        all_results = []

        for dist in self.config.distance_thresholds:
            for time_th in self.config.time_thresholds:
                rule_name = f"{dist}m{time_th}m"
                output_dir = f"{output_base_dir}/{rule_name}"

                results = detect_hospital_companions(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    distance_threshold=dist,
                    time_threshold_minutes=time_th,
                    id_column='id',
                    latitude_column='lat',
                    longitude_column='lon',
                    time_column='datetime',
                    num_processes=self.config.num_processes,
                )

                all_results.extend(results)

        return {
            "step": "detect_hospital_companions",
            "total_files": len(all_results),
            "success_files": sum(1 for r in all_results if r['status'] == 'success'),
        }

    def step2_detect_home_companions(
        self,
        input_dir: Optional[str] = None,
        output_base_dir: Optional[str] = None,
    ) -> dict:
        """
        Step 2: Detect Residence Companions

        Identify patient pairs with nearby residences based on geographic distance, using Haversine formula to calculate spherical distance.
        This step relies only on spatial distance and does not consider time factors.

        Parameters:
            input_dir (Optional[str]): Residence coordinate data input directory, default is data/trajectories/residence
            output_base_dir (Optional[str]): Output result base directory, default is data/companion_pairs/2_residence

        Returns:
            dict: Processing result statistics, containing the following fields:
                - step (str): Step name
                - total_files (int): Total number of processed files
                - success_files (int): Number of successfully processed files

        Processing Logic:
            1. Iterate through all distance thresholds (time thresholds not used)
            2. Calculate distances between all patient residences
            3. Filter patient pairs with distance less than threshold
            4. Output nearby residence patient pair list
        """
        logger.info("Step 2: Detect residence companions")

        input_dir = input_dir or str(self.base_dir / "trajectories/residence")
        output_base_dir = output_base_dir or str(self.base_dir / "companion_pairs/2_residence")

        all_results = []

        for dist in self.config.distance_thresholds:
            output_dir = f"{output_base_dir}/{dist}m"

            results = detect_home_companions(
                input_dir=input_dir,
                output_dir=output_dir,
                distance_threshold=dist,
                lat_column='lat',
                lon_column='lon',
                num_processes=self.config.num_processes,
            )

            all_results.extend(results)

        return {
            "step": "detect_home_companions",
            "total_files": len(all_results),
            "success_files": sum(1 for r in all_results if r['status'] == 'success'),
        }

    def step3_match_pairs(
        self,
        home_base_dir: Optional[str] = None,
        hospital_base_dir: Optional[str] = None,
        output_base_dir: Optional[str] = None,
    ) -> dict:
        """
        Step 3: Match Home and Hospital ABID Pairs

        Cross-match patient pairs detected in Step 1 and Step 2, filtering out patient pairs
        that simultaneously satisfy hospital contact and residence proximity conditions.

        Parameters:
            home_base_dir (Optional[str]): Residence companion data directory, default is data/companion_pairs/2_residence
            hospital_base_dir (Optional[str]): Hospital companion data directory, default is data/companion_pairs/1_hospital
            output_base_dir (Optional[str]): Output result base directory, default is data/companion_pairs/3_residence_hospital_matched

        Returns:
            dict: Processing result statistics, containing the following fields:
                - step (str): Step name
                - total_tasks (int): Total number of matching tasks
                - total_matched (int): Total number of successfully matched patient pairs

        Processing Logic:
            1. Extract distance threshold corresponding to each rule
            2. Read corresponding residence and hospital companion files
            3. Execute ABID pair intersection matching
            4. Output patient pairs satisfying both conditions
        """
        logger.info("Step 3: Match ABID pairs")

        home_base_dir = home_base_dir or str(self.base_dir / "companion_pairs/2_residence")
        hospital_base_dir = hospital_base_dir or str(self.base_dir / "companion_pairs/1_hospital")
        output_base_dir = output_base_dir or str(self.base_dir / "companion_pairs/3_residence_hospital_matched")

        all_results = []

        for rule_name in self.config.rule_combinations:
            # Extract distance threshold
            dist = int(rule_name.split('m')[0])

            home_folder = f"{home_base_dir}/{dist}m"
            hospital_folder = f"{hospital_base_dir}/{rule_name}"
            output_folder = f"{output_base_dir}/{rule_name}"

            results = match_home_hospital_pairs(
                home_folder=home_folder,
                hospital_folder=hospital_folder,
                output_folder=output_folder,
            )

            all_results.extend(results)

        return {
            "step": "match_pairs",
            "total_tasks": len(all_results),
            "total_matched": sum(r.matched_count for r in all_results),
        }

    def step4_extract_records(
        self,
        trajectory_dir: Optional[str] = None,
        accompany_base_dir: Optional[str] = None,
        output_base_dir: Optional[str] = None,
    ) -> dict:
        """
        Step 4: Extract All Trajectory Records of Companion Patients

        Extract complete spatiotemporal trajectory records of all patient pairs matched in Step 3
        from raw trajectory data, providing data foundation for subsequent journey contact analysis.

        Parameters:
            trajectory_dir (Optional[str]): Raw trajectory data directory, default is data/trajectories/origin_trajectories
            accompany_base_dir (Optional[str]): Matched patient pair data directory, default is data/companion_pairs/3_residence_hospital_matched
            output_base_dir (Optional[str]): Output result base directory, default is data/trajectories/residence_hospital_matched_all_trajectories

        Returns:
            dict: Processing result statistics, containing the following fields:
                - step (str): Step name
                - total_files (int): Total number of processed files
                - success_files (int): Number of successfully extracted files

        Processing Logic:
            1. Read matched patient pair files, extract unique patient ID list
            2. Scan raw trajectory data, match patient IDs
            3. Extract all trajectory records of matched patients
            4. Organize output files by city-date
        """
        logger.info("Step 4: Extract companion patient records")

        trajectory_dir = trajectory_dir or str(self.base_dir / "trajectories/origin_trajectories")
        accompany_base_dir = accompany_base_dir or str(self.base_dir / "companion_pairs/3_residence_hospital_matched")
        output_base_dir = output_base_dir or str(self.base_dir / "trajectories/residence_hospital_matched_all_trajectories")

        all_results = []

        for rule_name in self.config.rule_combinations:
            accompany_dir = f"{accompany_base_dir}/{rule_name}"
            output_dir = f"{output_base_dir}/{rule_name}"

            results = extract_companion_records(
                trajectory_dir=trajectory_dir,
                accompany_dir=accompany_dir,
                output_dir=output_dir,
                workers=self.config.num_processes,
            )

            all_results.extend(results)

        return {
            "step": "extract_records",
            "total_files": len(all_results),
            "success_files": sum(1 for r in all_results if r.success),
        }

    def step5_detect_contacts(
        self,
        accompany_base_dir: Optional[str] = None,
        trajectory_base_dir: Optional[str] = None,
        hospital_trajectory_dir: Optional[str] = None,
        home_coords_dir: Optional[str] = None,
        output_base_dir: Optional[str] = None,
    ) -> dict:
        """
        Step 5: Detect Journey Contacts

        Analyze spatiotemporal contact situations of patient pairs outside hospital and non-residential areas,
        comprehensively evaluating the overall contact frequency of patient pairs.
        This step excludes known hospital and residence contacts, focusing on contact analysis in other scenarios.

        Parameters:
            accompany_base_dir (Optional[str]): Matched patient pair data directory, default is data/companion_pairs/3_residence_hospital_matched
            trajectory_base_dir (Optional[str]): Extracted trajectory data directory, default is data/trajectories/residence_hospital_matched_all_trajectories
            hospital_trajectory_dir (Optional[str]): Hospital trajectory data directory, default is data/trajectories/hospital_trajectories
            home_coords_dir (Optional[str]): Residence coordinate data directory, default is data/trajectories/residence
            output_base_dir (Optional[str]): Output result base directory, default is data/companion_pairs/4_final_pairs

        Returns:
            dict: Processing result statistics, containing the following fields:
                - step (str): Step name
                - total_tasks (int): Total number of detection tasks
                - success_tasks (int): Number of successfully completed tasks

        Processing Logic:
            1. Load complete trajectory data of patient pairs
            2. Exclude trajectory points within hospital range (based on hospital trajectory data)
            3. Exclude trajectory points within residence range (based on home_radius configuration)
            4. Perform spatiotemporal contact detection on remaining trajectory points
            5. Output list of patient pairs with journey contacts
        """
        logger.info("Step 5: Detect journey contacts")

        accompany_base_dir = accompany_base_dir or str(self.base_dir / "companion_pairs/3_residence_hospital_matched")
        trajectory_base_dir = trajectory_base_dir or str(self.base_dir / "trajectories/residence_hospital_matched_all_trajectories")
        hospital_trajectory_dir = hospital_trajectory_dir or str(self.base_dir / "trajectories/hospital_trajectories")
        home_coords_dir = home_coords_dir or str(self.base_dir / "trajectories/residence")
        output_base_dir = output_base_dir or str(self.base_dir / "companion_pairs/4_final_pairs")

        all_results = []

        for rule_name in self.config.rule_combinations:
            ab_pairs_dir = f"{accompany_base_dir}/{rule_name}"
            full_trajectory_dir = f"{trajectory_base_dir}/{rule_name}"
            output_dir = f"{output_base_dir}/{rule_name}"

            results = detect_journey_contacts(
                ab_pairs_dir=ab_pairs_dir,
                full_trajectory_dir=full_trajectory_dir,
                hospital_trajectory_dir=hospital_trajectory_dir,
                home_coords_dir=home_coords_dir,
                output_dir=output_dir,
                home_radius=self.config.home_radius,
            )

            all_results.extend(results)

        return {
            "step": "detect_contacts",
            "total_tasks": len(all_results),
            "success_tasks": sum(1 for r in all_results if r.status == 'success'),
        }

    def step6_calculate_ratio(
        self,
        contact_base_dir: Optional[str] = None,
        patients_dir: Optional[str] = None,
        output_base_dir: Optional[str] = None,
    ) -> dict:
        """
        Step 6: Companion Classification and Ratio Calculation

        Build patient contact network based on NetworkX graph theory library, identify patient roles
        through network topology analysis, automatically classify patients and companion patients,
        and calculate companion patient ratios for each city-date.

        Parameters:
            contact_base_dir (Optional[str]): Journey contact patient pair data directory, default is data/companion_pairs/4_final_pairs
            patients_dir (Optional[str]): Unique patient list directory, default is data/trajectories/unique_patients
            output_base_dir (Optional[str]): Output result base directory, default is data/companions

        Returns:
            dict: Processing result statistics, containing the following fields:
                - step (str): Step name
                - total_tasks (int): Total number of calculation tasks

        Processing Logic:
            1. Build patient contact network graph (nodes = patient IDs, edges = contact relationships)
            2. Calculate network metrics: degree centrality, betweenness centrality, eigenvector centrality, clustering coefficient, network density
            3. Identify network structure types:
               - PAIR: Binary pair (2 nodes)
               - STAR: Star structure (central node + leaf nodes)
               - DENSE: Dense network (high clustering coefficient, high density)
               - MIXED: Mixed structure
            4. Classify patient roles based on network structure and node metrics:
               - Patients: Usually network central nodes or high-influence nodes
               - Companion patients: Usually peripheral nodes or low-influence nodes
            5. Calculate ratios
            6. Output results:
               - ratio.csv: Companion ratio statistics for each city-date
               - {date}/{city}/patients/: Patient ID list
               - {date}/{city}/companions/: Companion ID list
        """
        logger.info("Step 6: Calculate companion ratio")

        contact_base_dir = contact_base_dir or str(self.base_dir / "companion_pairs/4_final_pairs")
        patients_dir = patients_dir or str(self.base_dir / "trajectories/unique_patients")
        output_base_dir = output_base_dir or str(self.base_dir / "companions")

        all_results = []

        for rule_name in self.config.rule_combinations:
            ab_pairs_folder = f"{contact_base_dir}/{rule_name}"
            output_folder = f"{output_base_dir}/{rule_name}"

            results = calculate_companion_ratio(
                ab_pairs_folder=ab_pairs_folder,
                patients_folder=patients_dir,
                output_folder=output_folder,
            )

            all_results.extend(results)

        return {
            "step": "calculate_ratio",
            "total_tasks": len(all_results),
        }

    def _cleanup_intermediate_files(self):
        """
        Clean up intermediate files (internal method)

        Delete intermediate files generated during pipeline processing.
        This method will delete output files from steps 1-5, retaining final results (step 6 output).

        Cleanup Scope:
            - data/companion_pairs/1_hospital/
            - data/companion_pairs/2_residence/
            - data/companion_pairs/3_residence_hospital_matched/
            - data/trajectories/residence_hospital_matched_all_trajectories/
            - data/companion_pairs/4_final_pairs/

        Note:
            - Cleanup operation is irreversible
            - If cleanup fails, only warning log is recorded, pipeline execution continues
        """
        cleanup_dirs = [
            self.base_dir / "companion_pairs/1_hospital",
            self.base_dir / "companion_pairs/2_residence",
            self.base_dir / "companion_pairs/3_residence_hospital_matched",
            self.base_dir / "trajectories/residence_hospital_matched_all_trajectories",
            self.base_dir / "companion_pairs/4_final_pairs",
        ]

        for dir_path in cleanup_dirs:
            if dir_path.exists():
                try:
                    shutil.rmtree(dir_path)
                    logger.info(f"Cleaned intermediate files: {dir_path}")
                except Exception as e:
                    logger.warning(f"Cleanup failed {dir_path}: {e}")

    def run_full_pipeline(self, cleanup_intermediate: bool = True) -> dict:
        """
        Run complete six-step pipeline processing

        Parameters:
            cleanup_intermediate (bool): Whether to clean up intermediate files, default True
                - True: Clean up intermediate results in stages after steps 3, 5, and 6 completion
                - False: Retain all intermediate results for debugging and verification

        Returns:
            dict: Contains processing result statistics for all steps, keys are step names (step1-step6)

        Execution Flow:
            1. Execute step 1: Hospital companion detection
            2. Execute step 2: Residence companion detection
            3. Execute step 3: ABID pair matching
               └─> [Optional cleanup] Delete outputs of steps 1, 2
            4. Execute step 4: Trajectory record extraction
            5. Execute step 5: Journey contact detection
               └─> [Optional cleanup] Delete outputs of steps 3, 4
            6. Execute step 6: Companion ratio calculation
               └─> [Optional cleanup] Delete output of step 5
            7. Return statistical results of all steps

        Final Output Location:
            data/companions/{rule}/

        Usage Example:
            >>> config = PipelineConfig()
            >>> pipeline = CompanionAnalysisPipeline(config)
            >>> results = pipeline.run_full_pipeline(cleanup_intermediate=True)
            >>> for step, result in results.items():
            >>>     print(f"{step}: {result}")
        """
        logger.info("Starting complete pipeline processing")

        results = {}

        results["step1"] = self.step1_detect_hospital_companions()
        results["step2"] = self.step2_detect_home_companions()
        results["step3"] = self.step3_match_pairs()

        if cleanup_intermediate:
            for dir_path in [self.base_dir / "companion_pairs/1_hospital", self.base_dir / "companion_pairs/2_residence"]:
                if dir_path.exists():
                    try:
                        shutil.rmtree(dir_path)
                    except Exception as e:
                        logger.warning(f"Cleanup failed {dir_path}: {e}")

        results["step4"] = self.step4_extract_records()
        results["step5"] = self.step5_detect_contacts()

        if cleanup_intermediate:
            for dir_path in [self.base_dir / "companion_pairs/3_residence_hospital_matched", self.base_dir / "trajectories/residence_hospital_matched_all_trajectories"]:
                if dir_path.exists():
                    try:
                        shutil.rmtree(dir_path)
                    except Exception as e:
                        logger.warning(f"Cleanup failed {dir_path}: {e}")

        results["step6"] = self.step6_calculate_ratio()

        if cleanup_intermediate:
            dir_path = self.base_dir / "companion_pairs/4_final_pairs"
            if dir_path.exists():
                try:
                    shutil.rmtree(dir_path)
                except Exception as e:
                    logger.warning(f"Cleanup failed {dir_path}: {e}")

        logger.info(f"Pipeline processing completed, final results: {self.base_dir / 'companions'}")

        return results


def main():
    """
    Main function entry point

    Configure and start the complete companion patient analysis pipeline.
    This function sets up multiprocessing startup method, initializes configuration parameters,
    and executes the complete pipeline.

    Configuration Description:
        - base_data_dir: Root data directory
        - distance_thresholds: List of distance thresholds (meters)
        - time_thresholds: List of time thresholds (minutes)
        - num_processes: Number of parallel processes
        - home_radius: Residence exclusion radius (meters)
        - cleanup_intermediate: Whether to clean up intermediate files

    Notes:
        - Recommended to use 'spawn' startup
    """
    # Set multiprocessing startup method
    mp.set_start_method('spawn', force=True)

    # Configure pipeline parameters
    config = PipelineConfig(
        base_data_dir="data",
        distance_thresholds=[50],   # Distance threshold
        time_thresholds=[30],       # Time threshold
        num_processes=6,            # Number of parallel processes
        home_radius=300.0,          # Residence exclusion radius
    )

    # Initialize and run pipeline
    pipeline = CompanionAnalysisPipeline(config)
    results = pipeline.run_full_pipeline(cleanup_intermediate=True)

    # Output processing results for each step
    for step, result in results.items():
        logger.info(f"{step}: {result}")


if __name__ == "__main__":
    main()
