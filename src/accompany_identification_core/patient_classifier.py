# -*- coding: utf-8 -*-
"""
Companion Ratio Calculation Module

This module implements patient role identification and companion ratio calculation based on network graph theory,
serving as the core algorithm module of the companion patient analysis system.
By constructing patient contact networks and analyzing their topological structure, it automatically identifies
patient and companion patient roles.

Main Features:
==============
1. Network Construction
   - Build undirected graphs based on patient pair data
   - Nodes: Patient IDs
   - Edges: Contact relationships

2. Network Metrics Calculation
   - Degree Centrality: Number of connections per node
   - Betweenness Centrality: Importance of nodes in shortest paths
   - Eigenvector Centrality: Centrality considering neighbor importance
   - Clustering Coefficient: Connection density among neighbors
   - Network Density: Overall connection tightness

3. Network Structure Identification
   - PAIR: 2 nodes, 1 edge
   - STAR: Central node connects to multiple peripheral nodes, no connections among peripheral nodes
   - DENSE: High clustering coefficient, high density, similar node degrees
   - MIXED: Transitional structure with both star and dense characteristics
   - UNKNOWN: Structure that cannot be clearly classified

4. Patient Role Classification
   - Patient: Core figure in the network, usually disease spreader
   - Companion: Personnel in contact with patients, usually caregivers, family members, etc.

5. Companion Ratio Calculation
   - Calculate companion patient ratio for each city-date
   - Ratio = Number of companion patients / Total number of people

Data Flow:
==========
Input:
  Patient contact pair data (Parquet): data/final_pairs/{city}_{date}_contacts.parquet
  Fields: AID, BID, ...

Processing Flow:
  1. Read patient pair data
  2. Build NetworkX undirected graph
  3. Calculate network metrics
  4. Identify network structure types
  5. Execute role classification based on structure types
  6. Calculate companion patient ratio
  7. Save results

Output:
  ├─ ratio.csv: Companion ratio statistics for each city-date
  │   Fields: city, date, total_count, patient_count, companion_count, companion_ratio, structure_type
  └─ {date}/{city}/
      ├─ patients/: Patient ID list files
      └─ companions/: Companion patient ID list files


Usage Example:
=============
>>> from core.patient_classifier import calculate_companion_ratio
>>> results = calculate_companion_ratio(
...     ab_pairs_folder='data/final_pairs',
...     patients_folder='data/unique_patients',
...     output_folder='data/companions'
... )
>>> for result in results:
...     print(f"{result.city} {result.date}: "
...           f"Companion ratio={result.companion_ratio:.2%}, "
...           f"Structure type={result.structure_type}")

"""

import random
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set

import networkx as nx
import numpy as np
import pandas as pd


# Random seed
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class ComponentType(Enum):
    """Connected component type"""
    PAIR = auto()  # Pair structure
    STAR = auto()  # Star structure
    DENSE = auto()  # Dense structure
    MIXED = auto()  # Mixed structure
    UNKNOWN = auto()  # Cannot determine


@dataclass(frozen=True)
class ThresholdConfig:
    """Structure discrimination threshold configuration"""
    # Star structure thresholds
    star_clustering_max: float = 0.4
    star_density_max: float = 0.5
    star_min_degree: int = 3
    star_center_clustering_max: float = 0.3
    star_outer_clustering_min: float = 0.5

    # Dense structure thresholds
    dense_clustering_min: float = 0.5
    dense_density_min: float = 0.5
    dense_degree_ratio_max: float = 1.5

    # Mixed structure thresholds
    mixed_clustering_range: tuple = (0.3, 0.5)
    mixed_density_range: tuple = (0.3, 0.5)
    mixed_degree_ratio_range: tuple = (1.5, 3.0)


@dataclass(frozen=True)
class RoleScoreWeights:
    """Role scoring weights"""
    degree: float = 0.6
    betweenness: float = 0.3
    eigenvector: float = 0.1


@dataclass
class NetworkMetrics:
    """Network analysis metrics"""
    graph: nx.Graph
    degrees: Dict[str, int]
    betweenness: Dict[str, float]
    eigenvector: Dict[str, float]
    clustering: Dict[str, float]
    average_clustering: float
    density: float
    connected_components: List[Set[str]]

    @property
    def total_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def total_edges(self) -> int:
        return self.graph.number_of_edges()

    @property
    def component_count(self) -> int:
        return len(self.connected_components)


@dataclass
class RoleClassification:
    """Role classification results"""
    patients: Set[str]
    companions: Set[str]
    role_scores: Dict[str, float]
    component_types: Dict[FrozenSet[str], ComponentType]

    @property
    def companion_ratio(self) -> float:
        total = len(self.patients) + len(self.companions)
        return len(self.companions) / total if total > 0 else 0.0


@dataclass
class CalculationResult:
    """Calculation results"""
    city: str
    date: str
    network_nodes: int
    network_edges: int
    component_count: int
    identified_patients: int
    identified_companions: int
    patients_with_companions_ratio: float
    original_patients_count: int
    companion_ratio: float
    network_coverage: float
    avg_degree: float

    def to_dict(self) -> Dict:
        return {
            "city": self.city,
            "date": self.date,
            "identified_patients": self.identified_patients,
            "identified_companions": self.identified_companions,
            "patient_ratio": self.patients_with_companions_ratio,
            "companion_ratio": self.companion_ratio
        }


class NetworkAnalyzer:
    """Network structure analyzer"""

    def __init__(self, thresholds: ThresholdConfig = ThresholdConfig()):
        self.thresholds = thresholds

    def analyze(self, contact_df: pd.DataFrame) -> NetworkMetrics:
        """Analyze contact network structure"""
        graph = self._build_graph(contact_df)
        degrees = dict(graph.degree())
        betweenness = nx.betweenness_centrality(graph)
        eigenvector = self._compute_eigenvector_centrality(graph, degrees)
        clustering = nx.clustering(graph)

        return NetworkMetrics(
            graph=graph,
            degrees=degrees,
            betweenness=betweenness,
            eigenvector=eigenvector,
            clustering=clustering,
            average_clustering=nx.average_clustering(graph),
            density=nx.density(graph),
            connected_components=list(nx.connected_components(graph)),
        )

    @staticmethod
    def _build_graph(contact_df: pd.DataFrame) -> nx.Graph:
        """Build network graph from contact data"""
        graph = nx.Graph()
        edges = {
            tuple(sorted([row["AID"], row["BID"]])) for _, row in contact_df.iterrows()
        }
        graph.add_edges_from(edges)
        return graph

    @staticmethod
    def _compute_eigenvector_centrality(
        graph: nx.Graph, degrees: Dict[str, int]
    ) -> Dict[str, float]:
        """Calculate eigenvector centrality"""
        try:
            return nx.eigenvector_centrality(graph, max_iter=2000)
        except nx.PowerIterationFailedConvergence:
            max_degree = max(degrees.values()) if degrees else 1
            return {node: degree / max_degree for node, degree in degrees.items()}


class RoleClassifier:
    """Role classifier"""

    def __init__(
        self,
        thresholds: ThresholdConfig = ThresholdConfig(),
        weights: RoleScoreWeights = RoleScoreWeights(),
    ):
        self.thresholds = thresholds
        self.weights = weights

    def classify(self, network: NetworkMetrics) -> RoleClassification:
        """Perform role classification on nodes in the network"""
        role_scores = self._compute_role_scores(network)
        patients: Set[str] = set()
        companions: Set[str] = set()
        component_types: Dict[FrozenSet[str], ComponentType] = {}

        for component in network.connected_components:
            comp_key = frozenset(component)
            nodes = list(component)

            if len(nodes) == 2:
                patient, companion = self._classify_pair(nodes)
                patients.add(patient)
                companions.add(companion)
                component_types[comp_key] = ComponentType.PAIR

            elif len(nodes) >= 3:
                subgraph = network.graph.subgraph(component)
                patient, comp_companions, comp_type = self._classify_complex(subgraph)
                patients.add(patient)
                companions.update(comp_companions)
                component_types[comp_key] = comp_type

        return RoleClassification(
            patients=patients,
            companions=companions,
            role_scores=role_scores,
            component_types=component_types,
        )

    def _compute_role_scores(self, network: NetworkMetrics) -> Dict[str, float]:
        """Calculate node role scores"""
        nodes = list(network.graph.nodes())
        max_degree = max(network.degrees.values()) if network.degrees else 1

        scores = {}
        for node in nodes:
            degree_score = network.degrees[node] / max_degree
            scores[node] = (
                degree_score * self.weights.degree
                + network.betweenness[node] * self.weights.betweenness
                + network.eigenvector[node] * self.weights.eigenvector
            )
        return scores

    @staticmethod
    def _classify_pair(nodes: List[str]) -> tuple:
        """Classify pair structure"""
        patient = random.choice(nodes)
        companion = nodes[1] if nodes[0] == patient else nodes[0]
        return patient, companion

    def _classify_complex(self, subgraph: nx.Graph) -> tuple:
        """Classify complex structure"""
        degrees = dict(subgraph.degree())
        clustering = nx.clustering(subgraph)
        avg_clustering = nx.average_clustering(subgraph)
        density = nx.density(subgraph)
        nodes = list(subgraph.nodes())

        degree_values = list(degrees.values())
        max_degree, min_degree = max(degree_values), min(degree_values)
        avg_degree = np.mean(degree_values)
        degree_ratio = max_degree / min_degree if min_degree > 0 else float("inf")

        # Identify various structures
        result = self._try_star_structure(
            nodes, degrees, clustering, avg_clustering, density, max_degree, avg_degree
        )
        if result:
            return result

        result = self._try_dense_structure(
            nodes, degrees, avg_clustering, density, degree_ratio
        )
        if result:
            return result

        result = self._try_mixed_structure(
            nodes, clustering, avg_clustering, density, degree_ratio
        )
        if result:
            return result

        # Cannot determine
        patient = random.choice(nodes)
        return patient, set(nodes) - {patient}, ComponentType.UNKNOWN

    def _try_star_structure(
        self, nodes, degrees, clustering, avg_clustering, density, max_degree, avg_degree
    ) -> Optional[tuple]:
        """Identify star structure"""
        th = self.thresholds

        if not (
            avg_clustering < th.star_clustering_max
            and density < th.star_density_max
            and max_degree >= th.star_min_degree
            and max_degree > avg_degree + 1
        ):
            return None

        center_candidates = [n for n in nodes if degrees[n] == max_degree]

        for center in center_candidates:
            if clustering.get(center, 0) > th.star_center_clustering_max:
                continue

            outer_nodes = set(nodes) - {center}
            outer_clustering = [clustering.get(n, 0) for n in outer_nodes]
            avg_outer = np.mean(outer_clustering) if outer_clustering else 0

            if avg_outer >= th.star_outer_clustering_min:
                return center, outer_nodes, ComponentType.STAR

        return None

    def _try_dense_structure(
        self, nodes, degrees, avg_clustering, density, degree_ratio
    ) -> Optional[tuple]:
        """Identify dense structure"""
        th = self.thresholds

        if not (
            avg_clustering > th.dense_clustering_min
            and density > th.dense_density_min
            and degree_ratio < th.dense_degree_ratio_max
        ):
            return None

        max_degree = max(degrees.values())
        candidates = [n for n in nodes if degrees[n] == max_degree]
        patient = candidates[0] if len(candidates) == 1 else random.choice(candidates)

        return patient, set(nodes) - {patient}, ComponentType.DENSE

    def _try_mixed_structure(
        self, nodes, clustering, avg_clustering, density, degree_ratio
    ) -> Optional[tuple]:
        """Identify mixed structure"""
        th = self.thresholds
        clust_min, clust_max = th.mixed_clustering_range
        dens_min, dens_max = th.mixed_density_range
        ratio_min, ratio_max = th.mixed_degree_ratio_range

        if not (
            clust_min <= avg_clustering <= clust_max
            and dens_min <= density <= dens_max
            and ratio_min <= degree_ratio <= ratio_max
        ):
            return None

        patient = min(nodes, key=lambda n: clustering.get(n, 1))
        return patient, set(nodes) - {patient}, ComponentType.MIXED


class RatioCalculator:
    """Companion ratio calculator"""

    AB_PAIRS_SUFFIX = "_final_companion.parquet"
    PATIENTS_SUFFIX = "_unique_id.parquet"

    def __init__(self, ab_pairs_folder: str, patients_folder: str):
        self.ab_pairs_folder = Path(ab_pairs_folder)
        self.patients_folder = Path(patients_folder)
        self.network_analyzer = NetworkAnalyzer()
        self.role_classifier = RoleClassifier()
        self.results: List[CalculationResult] = []

    def find_file_mappings(self) -> List[Dict]:
        """Find file mappings"""
        mappings = []
        ab_files = list(self.ab_pairs_folder.glob("*.parquet"))

        for ab_file in ab_files:
            mapping = self._parse_ab_file(ab_file)
            if mapping:
                mappings.append(mapping)

        return mappings

    def _parse_ab_file(self, ab_file: Path) -> Optional[Dict]:
        """Parse AB pairs file"""
        filename = ab_file.name

        if not filename.endswith(self.AB_PAIRS_SUFFIX):
            return None

        base_name = filename.replace(self.AB_PAIRS_SUFFIX, "")
        parts = base_name.split("_", 1)

        if len(parts) < 2:
            return None

        city, date = parts[0], parts[1]
        patients_file = (
            self.patients_folder / city / f"{city}_{date}{self.PATIENTS_SUFFIX}"
        )

        if not patients_file.exists():
            return None

        return {
            "city": city,
            "date": date,
            "ab_pairs_file": ab_file,
            "patients_file": patients_file,
        }

    def process_single(self, mapping: Dict) -> Optional[CalculationResult]:
        """Process single city-date"""
        # Read data
        try:
            contact_df = pd.read_parquet(mapping["ab_pairs_file"])
            patients_df = pd.read_parquet(mapping["patients_file"])
        except Exception:
            return None

        if len(contact_df) == 0:
            return None

        # Analyze network
        network = self.network_analyzer.analyze(contact_df)

        # Classify roles
        roles = self.role_classifier.classify(network)

        # Save role details
        self._store_role_details(mapping["city"], mapping["date"], roles)

        # Calculate statistics
        all_ids = set(contact_df["AID"]).union(set(contact_df["BID"]))
        original_patients_count = len(patients_df)

        patients_with_companions_ratio = (
            len(roles.patients) / original_patients_count * 100
            if original_patients_count > 0 else 0
        )

        companion_ratio = (
            len(roles.companions) / original_patients_count * 100
            if original_patients_count > 0 else 0
        )

        network_coverage = (
            len(all_ids) / original_patients_count * 100
            if original_patients_count > 0 else 0
        )

        degree_values = list(network.degrees.values())

        return CalculationResult(
            city=mapping["city"],
            date=mapping["date"],
            network_nodes=network.total_nodes,
            network_edges=network.total_edges,
            component_count=network.component_count,
            identified_patients=len(roles.patients),
            identified_companions=len(roles.companions),
            patients_with_companions_ratio=round(patients_with_companions_ratio, 2),
            original_patients_count=original_patients_count,
            companion_ratio=round(companion_ratio, 2),
            network_coverage=round(network_coverage, 2),
            avg_degree=round(np.mean(degree_values), 2) if degree_values else 0,
        )

    def _store_role_details(self, city: str, date: str, roles: RoleClassification):
        """Store role details"""
        if not hasattr(self, "role_details"):
            self.role_details = []

        for pid in roles.patients:
            self.role_details.append({"city": city, "date": date, "id": pid, "role": "patient"})
        for cid in roles.companions:
            self.role_details.append({"city": city, "date": date, "id": cid, "role": "companion"})

    def process_all(self, progress: bool = True) -> List[CalculationResult]:
        """Process all data"""
        from tqdm import tqdm

        mappings = self.find_file_mappings()
        self.role_details = []  # Reset details

        results = []
        iterator = tqdm(mappings, desc="Calculating companion ratios") if progress else mappings

        for mapping in iterator:
            result = self.process_single(mapping)
            if result:
                results.append(result)

        self.results = results
        return results

    def save_results(self, output_folder: str) -> None:
        """Save results"""
        if not self.results:
            return

        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        # Detailed results
        df = pd.DataFrame([r.to_dict() for r in self.results])
        df = df.sort_values(["city", "date"])
        df.to_csv(output_path / "ratio.csv", index=False, encoding="utf-8")

        # Save role lists to corresponding folders
        if hasattr(self, "role_details") and self.role_details:
            # Group by city and date
            grouped = {}
            for item in self.role_details:
                key = (item["city"], item["date"])
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(item)

            # Create corresponding folders for each city-date
            for (city, date), items in grouped.items():
                # Create directory structure
                date_path = output_path / date / city

                # Patients folder
                patients_path = date_path / "patients"
                patients_path.mkdir(parents=True, exist_ok=True)
                patients_df = pd.DataFrame([{"ID": item["id"], "Role": "patient"}
                                          for item in items if item["role"] == "patient"])
                if not patients_df.empty:
                    patients_df.to_csv(patients_path / f"{city}_{date}_patients.csv",
                                      index=False, encoding="utf-8")

                # Companions folder
                companions_path = date_path / "companions"
                companions_path.mkdir(parents=True, exist_ok=True)
                companions_df = pd.DataFrame([{"ID": item["id"], "Role": "companion"}
                                            for item in items if item["role"] == "companion"])
                if not companions_df.empty:
                    companions_df.to_csv(companions_path / f"{city}_{date}_companions.csv",
                                        index=False, encoding="utf-8")


def calculate_companion_ratio(
    ab_pairs_folder: str,
    patients_folder: str,
    output_folder: str,
    progress: bool = True
) -> List[CalculationResult]:
    """Calculate companion ratio"""
    calculator = RatioCalculator(ab_pairs_folder, patients_folder)
    results = calculator.process_all(progress)

    if results:
        calculator.save_results(output_folder)

    return results
