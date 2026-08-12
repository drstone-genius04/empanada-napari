"""Mitochondrial morphometrics for labeled instance segmentations.

Overview
--------
For each mitochondrion in a cell, this module quantifies:

- **Shape class**: tube, sheet, junction, fusion site, bulb, nanotunnel, donut
- **Structural complexity**: branching, curvature, length, tortuosity
- **Local abnormal features**: bulbs, nanotunnels, donuts
- **Population context**: contact area, neighbors, clustering (2D only)

The goal is to turn a 2D/3D label image into quantitative morphology phenotypes
that can be compared across cells, conditions, or time points — not just
"volume = X", but e.g. "60% tubular, 20% sheet-like, 2 bulbs, 1 nanotunnel,
3 junctions".

Structural decomposition
------------------------
Each mitochondrion is decomposed into biologically meaningful parts:

Tube
    Elongated tubular segment — classic networked mitochondrial morphology.
Sheet
    Flat / lamellar region — often associated with fused or expanded morphology.
Junction
    Branch point where skeleton paths meet — branching complexity.
Fusion (TLF)
    Tiny lateral fusion — short bridge-like connections between nearby branches.
Bulb
    Local swelling along a tube — diameter increase above local median.
Nanotunnel
    Very thin, long segment — radius below threshold for minimum length.
Donut
    Closed loop in the skeleton — ring-like mitochondrial morphology.

Measurement categories
----------------------
1. **Global organelle shape**
   Volume, surface area, sphericity, centroid, orientation, eccentricity.
   Captures overall size and compactness vs elongation.

2. **Composition fractions**
   ``count_*``, ``length_*``, ``volume_*``, ``fraction_*_volume`` for each
   structure type. Distinguishes network-like (tubes) vs flattened/fused (sheets).

3. **Tube geometry**
   Skeleton length, tortuosity, mean/max tube radius, branching angles.
   Describes how straight, bent, thick, and branched the network is.

4. **Local pathology-like features**
   Bulbs, nanotunnels, donuts — non-uniform morphology missed by volume alone.

5. **Network / population context (2D only)**
   ``num_neighbors``, ``contact_area_*``, ``clustering_coefficient``.
   Describes whether a mitochondrion is isolated or part of a local cluster.

Output
------
``analyze_labels`` and ``analyze_labels_stack`` return a pandas DataFrame with
one row per label ID and columns for the metrics above.

See Also
--------
empanada_napari._mito_metrics_widget : Napari GUI for running morphometrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.measure import regionprops, regionprops_table
from skimage.morphology import skeletonize

__all__ = [
    "MitoMorphometricsConfig",
    "analyze_instance",
    "analyze_labels",
    "analyze_labels_stack",
]

NEIGHBOR_OFFSETS_2D = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

NEIGHBOR_OFFSETS_3D = [
    (dz, dy, dx)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if not (dz == 0 and dy == 0 and dx == 0)
]

STRUCTURE_TYPES = (
    "tube",
    "sheet",
    "junction",
    "fusion",
    "bulb",
    "nanotunnel",
    "donut",
)


@dataclass
class MitoMorphometricsConfig:
    """Thresholds and physical scale for mitochondrial morphometrics.

    Attributes
    ----------
    pixel_size_nm
        XY pixel size in nanometers; converts pixel counts to physical units.
    z_step_nm
        Z spacing in nanometers for 3D volumes. Defaults to ``pixel_size_nm``.
    nanotunnel_max_radius_nm
        Maximum tube radius (nm) for classifying a segment as a nanotunnel or
        tiny lateral fusion (TLF).
    nanotunnel_min_length_nm
        Minimum skeleton path length (nm) required to call a thin segment a
        nanotunnel rather than a fusion bridge.
    fusion_max_length_nm
        Maximum path length (nm) for classifying a thin segment as fusion (TLF).
    bulb_radius_ratio
        Local radius must exceed this multiple of the path median radius to
        classify a segment as a bulb (local swelling).
    sheet_solidity_threshold
        Minimum region solidity to prefer sheet over tube classification.
    sheet_elongation_threshold
        Maximum major/minor axis ratio to prefer sheet over tube classification.
    neighbor_contact_radius_nm
        Dilation radius (nm) for detecting contact between mitochondria in 2D
        network-context metrics.
    """

    pixel_size_nm: float = 1.0
    z_step_nm: Optional[float] = None
    nanotunnel_max_radius_nm: float = 100.0
    nanotunnel_min_length_nm: float = 150.0
    fusion_max_length_nm: float = 80.0
    bulb_radius_ratio: float = 1.5
    sheet_solidity_threshold: float = 0.85
    sheet_elongation_threshold: float = 2.5
    neighbor_contact_radius_nm: float = 20.0

    @property
    def voxel_volume_nm3(self) -> float:
        z = self.z_step_nm if self.z_step_nm is not None else self.pixel_size_nm
        return self.pixel_size_nm ** 2 * z

    @property
    def nanotunnel_max_radius_px(self) -> float:
        return self.nanotunnel_max_radius_nm / self.pixel_size_nm

    @property
    def nanotunnel_min_length_px(self) -> float:
        return self.nanotunnel_min_length_nm / self.pixel_size_nm

    @property
    def fusion_max_length_px(self) -> float:
        return self.fusion_max_length_nm / self.pixel_size_nm

    @property
    def neighbor_contact_radius_px(self) -> int:
        return max(1, int(round(self.neighbor_contact_radius_nm / self.pixel_size_nm)))


def _sphericity_2d(area: float, perimeter: float) -> float:
    if perimeter <= 0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter ** 2))


def _sphericity_3d(volume: float, surface_area: float) -> float:
    if surface_area <= 0:
        return 0.0
    return float((np.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0)) / surface_area)


def _surface_area_3d(mask: np.ndarray) -> float:
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    gx, gy, gz = np.gradient(padded.astype(float))
    return float(np.sum(np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)))


def _skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 2:
        return skeletonize(mask.astype(bool))
    # skimage>=0.22 no longer exposes skeletonize_3d; per-slice 2D
    # skeletonization is a practical fallback for EM z-stacks.
    sk = np.zeros_like(mask, dtype=bool)
    for i in range(mask.shape[0]):
        sl = mask[i].astype(bool)
        if sl.any():
            sk[i] = skeletonize(sl)
    return sk


def _neighbor_offsets(ndim: int) -> List[Tuple[int, ...]]:
    return NEIGHBOR_OFFSETS_2D if ndim == 2 else NEIGHBOR_OFFSETS_3D


def _build_skeleton_graph(skeleton: np.ndarray) -> nx.Graph:
    coords = np.column_stack(np.nonzero(skeleton))
    if coords.size == 0:
        return nx.Graph()

    index_map = {tuple(c): i for i, c in enumerate(coords)}
    offsets = _neighbor_offsets(skeleton.ndim)
    graph = nx.Graph()
    for i, coord in enumerate(coords):
        graph.add_node(i, coord=tuple(coord))
        for offset in offsets:
            neighbor = tuple(coord + np.array(offset))
            j = index_map.get(neighbor)
            if j is not None and i < j:
                dist = float(np.linalg.norm(np.array(offset, dtype=float)))
                graph.add_edge(i, j, weight=dist)

    return graph


def _edge_midpoints(graph: nx.Graph) -> Dict[Tuple[int, int], np.ndarray]:
    mids = {}
    for u, v, data in graph.edges(data=True):
        cu = np.array(graph.nodes[u]["coord"], dtype=float)
        cv = np.array(graph.nodes[v]["coord"], dtype=float)
        key = (min(u, v), max(u, v))
        mids[key] = 0.5 * (cu + cv)
    return mids


def _path_length(graph: nx.Graph, path: Sequence[int]) -> float:
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        if graph.has_edge(a, b):
            total += graph.edges[a, b].get("weight", 1.0)
        else:
            ca = np.array(graph.nodes[a]["coord"], dtype=float)
            cb = np.array(graph.nodes[b]["coord"], dtype=float)
            total += float(np.linalg.norm(ca - cb))
    return total


def _trace_skeleton_paths(graph: nx.Graph) -> List[List[int]]:
    if graph.number_of_nodes() == 0:
        return []

    branch_nodes = {n for n, d in graph.degree() if d != 2}
    if not branch_nodes:
        if graph.number_of_nodes() >= 2:
            nodes = list(graph.nodes)
            return [nodes]
        return [[n] for n in graph.nodes]

    visited_edges = set()
    paths: List[List[int]] = []

    def _edge_key(a: int, b: int) -> Tuple[int, int]:
        return (min(a, b), max(a, b))

    for start in branch_nodes:
        for neighbor in graph.neighbors(start):
            key = _edge_key(start, neighbor)
            if key in visited_edges:
                continue
            path = [start, neighbor]
            visited_edges.add(key)
            current = neighbor
            prev = start
            while graph.degree(current) == 2:
                nxt = [n for n in graph.neighbors(current) if n != prev][0]
                key = _edge_key(current, nxt)
                if key in visited_edges:
                    break
                visited_edges.add(key)
                path.append(nxt)
                prev, current = current, nxt
            paths.append(path)

    return paths


def _junction_angles(graph: nx.Graph, node: int) -> List[float]:
    neighbors = list(graph.neighbors(node))
    if len(neighbors) < 3:
        return []
    vectors = []
    center = np.array(graph.nodes[node]["coord"], dtype=float)
    for nb in neighbors:
        vec = np.array(graph.nodes[nb]["coord"], dtype=float) - center
        norm = np.linalg.norm(vec)
        if norm > 0:
            vectors.append(vec / norm)
    angles = []
    for v1, v2 in combinations(vectors, 2):
        dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
        angles.append(float(np.degrees(np.arccos(dot))))
    return angles


def _assign_structure_pixels(
    mask: np.ndarray,
    skeleton: np.ndarray,
    structure_map: np.ndarray,
) -> Dict[str, float]:
    if not mask.any():
        return {f"fraction_{name}_volume": 0.0 for name in STRUCTURE_TYPES}

    if skeleton.any():
        _, indices = ndimage.distance_transform_edt(~skeleton, return_indices=True)
        nearest_types = structure_map[tuple(indices)]
    else:
        nearest_types = np.full(mask.shape, "sheet", dtype=object)

    total = float(mask.sum())
    fractions = {}
    for name in STRUCTURE_TYPES:
        fractions[f"fraction_{name}_volume"] = float((nearest_types == name)[mask].sum() / total)
    return fractions


def _classify_paths_and_features(
    mask: np.ndarray,
    skeleton: np.ndarray,
    graph: nx.Graph,
    paths: List[List[int]],
    radii: np.ndarray,
    config: MitoMorphometricsConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
    structure_map = np.full(skeleton.shape, "", dtype=object)
    counts = {name: 0 for name in STRUCTURE_TYPES}
    lengths = {name: 0.0 for name in STRUCTURE_TYPES}
    volumes = {name: 0 for name in STRUCTURE_TYPES}
    branching_angles: List[float] = []

    # Closed skeleton loops (donuts)
    for cycle in nx.cycle_basis(graph):
        if len(cycle) >= 4:
            counts["donut"] += 1
            closed = cycle + [cycle[0]]
            lengths["donut"] += _path_length(graph, closed)
            for node in cycle:
                structure_map[tuple(graph.nodes[node]["coord"])] = "donut"

    for node, degree in graph.degree():
        if structure_map[tuple(graph.nodes[node]["coord"])] == "donut":
            continue
        if degree >= 3:
            structure_map[tuple(graph.nodes[node]["coord"])] = "junction"
            counts["junction"] += 1
            branching_angles.extend(_junction_angles(graph, node))

    for path in paths:
        if len(path) < 2:
            continue
        if any(structure_map[tuple(graph.nodes[n]["coord"])] == "donut" for n in path):
            continue
        path_len = _path_length(graph, path)
        path_radii = [radii[tuple(graph.nodes[n]["coord"])] for n in path]
        median_radius = float(np.median(path_radii)) if path_radii else 0.0
        mean_radius = float(np.mean(path_radii)) if path_radii else 0.0
        is_closed = path[0] == path[-1] or (
            graph.has_edge(path[0], path[-1]) and len(path) > 2
        )

        if is_closed and path_len > 0:
            structure_type = "donut"
        elif (
            mean_radius <= config.nanotunnel_max_radius_px
            and path_len >= config.nanotunnel_min_length_px
        ):
            structure_type = "nanotunnel"
        elif (
            mean_radius <= config.nanotunnel_max_radius_px
            and path_len <= config.fusion_max_length_px
        ):
            structure_type = "fusion"
        else:
            bulb_pixels = sum(
                1 for r in path_radii if median_radius > 0 and r >= median_radius * config.bulb_radius_ratio
            )
            if bulb_pixels >= max(2, len(path_radii) // 4):
                structure_type = "bulb"
            else:
                structure_type = "tube"

        counts[structure_type] += 1
        lengths[structure_type] += path_len
        for node in path:
            coord = tuple(graph.nodes[node]["coord"])
            if structure_map[coord] == "":
                structure_map[coord] = structure_type

    # Remaining skeleton pixels default to tube
    structure_map[(skeleton & (structure_map == ""))] = "tube"

    # Sheet assignment for non-skeleton mask pixels with high solidity / low elongation
    rp = regionprops(mask.astype(np.uint8))[0] if mask.any() else None
    elongation = 1.0
    solidity = 1.0
    if rp is not None:
        elongation = (
            rp.major_axis_length / max(rp.minor_axis_length, 1e-6)
            if hasattr(rp, "major_axis_length")
            else 1.0
        )
        solidity = float(getattr(rp, "solidity", 1.0))

    if (
        solidity >= config.sheet_solidity_threshold
        and elongation <= config.sheet_elongation_threshold
        and counts["tube"] == 0
        and counts["nanotunnel"] == 0
    ):
        for coord in np.column_stack(np.nonzero(skeleton)):
            c = tuple(coord)
            if structure_map[c] == "":
                structure_map[c] = "sheet"
        counts["sheet"] += 1

    fractions = _assign_structure_pixels(mask, skeleton, structure_map)
    for name in STRUCTURE_TYPES:
        volumes[name] = int(round(fractions[f"fraction_{name}_volume"] * mask.sum()))

    metrics = {
        **{f"count_{name}": counts[name] for name in STRUCTURE_TYPES},
        **{f"length_{name}_px": lengths[name] for name in STRUCTURE_TYPES},
        **{f"volume_{name}_px": volumes[name] for name in STRUCTURE_TYPES},
        **fractions,
        "mean_branching_angle_deg": float(np.mean(branching_angles)) if branching_angles else 0.0,
        "max_branching_angle_deg": float(np.max(branching_angles)) if branching_angles else 0.0,
        "solidity": solidity,
        "elongation": float(elongation),
    }
    return structure_map, metrics


def _global_metrics_2d(mask: np.ndarray, config: MitoMorphometricsConfig) -> Dict[str, float]:
    props = regionprops_table(
        mask.astype(np.uint8),
        properties=(
            "area",
            "perimeter",
            "equivalent_diameter",
            "major_axis_length",
            "minor_axis_length",
            "eccentricity",
            "orientation",
            "solidity",
            "extent",
        ),
    )
    area = float(props["area"][0])
    perimeter = float(props["perimeter"][0])
    centroid = regionprops(mask.astype(np.uint8))[0].centroid

    return {
        "volume_px": area,
        "volume_nm3": area * (config.pixel_size_nm ** 2),
        "surface_area_px": perimeter,
        "surface_area_nm2": perimeter * config.pixel_size_nm,
        "equivalent_diameter_px": float(props["equivalent_diameter"][0]),
        "equivalent_diameter_nm": float(props["equivalent_diameter"][0]) * config.pixel_size_nm,
        "major_axis_length_px": float(props["major_axis_length"][0]),
        "major_axis_length_nm": float(props["major_axis_length"][0]) * config.pixel_size_nm,
        "minor_axis_length_px": float(props["minor_axis_length"][0]),
        "minor_axis_length_nm": float(props["minor_axis_length"][0]) * config.pixel_size_nm,
        "eccentricity": float(props["eccentricity"][0]),
        "orientation_rad": float(props["orientation"][0]),
        "solidity": float(props["solidity"][0]),
        "extent": float(props["extent"][0]),
        "sphericity": _sphericity_2d(area, perimeter),
        "centroid_y": float(centroid[0]),
        "centroid_x": float(centroid[1]),
    }


def _global_metrics_3d(mask: np.ndarray, config: MitoMorphometricsConfig) -> Dict[str, float]:
    props = regionprops_table(
        mask.astype(np.uint8),
        properties=(
            "area",
            "equivalent_diameter",
            "major_axis_length",
            "minor_axis_length",
            "eccentricity",
            "orientation",
            "solidity",
            "extent",
        ),
    )
    volume = float(props["area"][0])
    surface_area = _surface_area_3d(mask)
    centroid = regionprops(mask.astype(np.uint8))[0].centroid
    z_scale = config.z_step_nm if config.z_step_nm is not None else config.pixel_size_nm

    return {
        "volume_px": volume,
        "volume_nm3": volume * config.voxel_volume_nm3,
        "surface_area_px": surface_area,
        "surface_area_nm2": surface_area * (config.pixel_size_nm ** 2),
        "equivalent_diameter_px": float(props["equivalent_diameter"][0]),
        "equivalent_diameter_nm": float(props["equivalent_diameter"][0]) * config.pixel_size_nm,
        "major_axis_length_px": float(props["major_axis_length"][0]),
        "major_axis_length_nm": float(props["major_axis_length"][0]) * config.pixel_size_nm,
        "minor_axis_length_px": float(props["minor_axis_length"][0]),
        "minor_axis_length_nm": float(props["minor_axis_length"][0]) * config.pixel_size_nm,
        "eccentricity": float(props["eccentricity"][0]),
        "orientation_rad": float(props["orientation"][0]),
        "solidity": float(props["solidity"][0]),
        "extent": float(props["extent"][0]),
        "sphericity": _sphericity_3d(volume, surface_area),
        "centroid_z": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "centroid_x": float(centroid[2]),
    }


def _tube_geometry_metrics(
    mask: np.ndarray,
    skeleton: np.ndarray,
    graph: nx.Graph,
    paths: List[List[int]],
    radii: np.ndarray,
    config: MitoMorphometricsConfig,
) -> Dict[str, float]:
    skeleton_length = float(skeleton.sum()) if skeleton.ndim == 2 else float(skeleton.sum())
    endpoint_nodes = [n for n, d in graph.degree() if d == 1]
    tortuosity = 0.0
    if len(endpoint_nodes) >= 2:
        try:
            path = nx.shortest_path(
                graph,
                endpoint_nodes[0],
                endpoint_nodes[-1],
                weight="weight",
            )
            geodesic = _path_length(graph, path)
            coords = np.array([graph.nodes[n]["coord"] for n in path], dtype=float)
            euclidean = float(np.linalg.norm(coords[0] - coords[-1]))
            tortuosity = geodesic / max(euclidean, 1e-6)
        except nx.NetworkXNoPath:
            tortuosity = 0.0

    skeleton_radii = radii[skeleton] if skeleton.any() else np.array([0.0])
    return {
        "skeleton_length_px": skeleton_length,
        "skeleton_length_nm": skeleton_length * config.pixel_size_nm,
        "tortuosity": tortuosity,
        "mean_tube_radius_px": float(np.mean(skeleton_radii)) if skeleton_radii.size else 0.0,
        "mean_tube_radius_nm": float(np.mean(skeleton_radii) * config.pixel_size_nm) if skeleton_radii.size else 0.0,
        "max_tube_radius_px": float(np.max(skeleton_radii)) if skeleton_radii.size else 0.0,
        "max_tube_radius_nm": float(np.max(skeleton_radii) * config.pixel_size_nm) if skeleton_radii.size else 0.0,
        "tube_radius_std_px": float(np.std(skeleton_radii)) if skeleton_radii.size else 0.0,
    }


def analyze_instance(
    mask: np.ndarray,
    label_id: int,
    config: Optional[MitoMorphometricsConfig] = None,
) -> Dict[str, float]:
    """Analyze a single binary mitochondrion mask.

    Returns global shape, structural decomposition, tube geometry, and feature
    counts/fractions for one labeled instance.
    """
    config = config or MitoMorphometricsConfig()
    mask = mask.astype(bool)
    if not mask.any():
        raise ValueError(f"Label {label_id} mask is empty.")

    # Crop to bounding box so EDT / skeletonization stay cheap.
    # Pad by 1 so EDT still sees exterior background (a tight crop that is
    # entirely True would otherwise treat array borders as the only "outside").
    coords = np.where(mask)
    slices = tuple(slice(int(c.min()), int(c.max()) + 1) for c in coords)
    cropped = np.pad(mask[slices], 1, mode="constant", constant_values=False)
    origin = [int(c.min()) - 1 for c in coords]

    skeleton = _skeletonize_mask(cropped)
    graph = _build_skeleton_graph(skeleton)
    paths = _trace_skeleton_paths(graph)
    radii = ndimage.distance_transform_edt(cropped)

    if cropped.ndim == 2:
        metrics = _global_metrics_2d(cropped, config)
    else:
        metrics = _global_metrics_3d(cropped, config)

    _, structure_metrics = _classify_paths_and_features(
        cropped, skeleton, graph, paths, radii, config
    )
    tube_metrics = _tube_geometry_metrics(cropped, skeleton, graph, paths, radii, config)

    metrics.update(structure_metrics)
    metrics.update(tube_metrics)
    metrics["label_id"] = int(label_id)
    metrics["dimension"] = int(mask.ndim)

    # Restore centroids to original image coordinates.
    if "centroid_z" in metrics:
        metrics["centroid_z"] += origin[0]
        metrics["centroid_y"] += origin[1]
        metrics["centroid_x"] += origin[2]
    else:
        metrics["centroid_y"] += origin[0]
        metrics["centroid_x"] += origin[1]
    return metrics


def _compute_network_context(
    labels: np.ndarray,
    label_ids: Sequence[int],
    config: MitoMorphometricsConfig,
) -> Dict[int, Dict[str, float]]:
    if labels.ndim != 2:
        return {int(label_id): {"num_neighbors": 0, "contact_area_px": 0.0, "clustering_coefficient": 0.0} for label_id in label_ids}

    radius = config.neighbor_contact_radius_px
    y, x = np.ogrid[-radius: radius + 1, -radius: radius + 1]
    footprint = x ** 2 + y ** 2 <= radius ** 2
    adjacency = {int(label_id): set() for label_id in label_ids}
    contact_area = {int(label_id): 0.0 for label_id in label_ids}

    for label_id in label_ids:
        dilated = ndimage.binary_dilation(labels == label_id, structure=footprint)
        touching = np.unique(labels[dilated & (labels != label_id)])
        touching = [int(v) for v in touching if v != 0]
        adjacency[int(label_id)] = set(touching)
        contact_area[int(label_id)] = float((dilated & np.isin(labels, touching)).sum())

    graph = nx.Graph()
    for label_id in label_ids:
        graph.add_node(int(label_id))
    for a, b in combinations(label_ids, 2):
        if b in adjacency[int(a)]:
            graph.add_edge(int(a), int(b))

    clustering = nx.clustering(graph) if graph.number_of_nodes() else {}
    results = {}
    for label_id in label_ids:
        lid = int(label_id)
        results[lid] = {
            "num_neighbors": len(adjacency[lid]),
            "contact_area_px": contact_area[lid],
            "contact_area_nm2": contact_area[lid] * (config.pixel_size_nm ** 2),
            "clustering_coefficient": float(clustering.get(lid, 0.0)),
        }
    return results


def analyze_labels(
    labels: np.ndarray,
    label_ids: Optional[Iterable[int]] = None,
    config: Optional[MitoMorphometricsConfig] = None,
    compute_network_context: bool = True,
) -> pd.DataFrame:
    """Analyze all instances in a 2D or 3D label image.

    Returns a DataFrame with one row per label ID. When ``compute_network_context``
    is True and the input is 2D, adds neighbor contact and clustering columns.
    """
    config = config or MitoMorphometricsConfig()
    labels = np.asarray(labels)
    if label_ids is None:
        label_ids = [int(v) for v in np.unique(labels) if v != 0]
    else:
        label_ids = [int(v) for v in label_ids if v != 0]

    n_labels = len(label_ids)
    print(f"Analyzing {n_labels} mitochondria in labels of shape {labels.shape}...")

    rows = []
    network = (
        _compute_network_context(labels, label_ids, config)
        if compute_network_context
        else {}
    )

    for i, label_id in enumerate(label_ids, start=1):
        mask = labels == label_id
        if not mask.any():
            continue
        row = analyze_instance(mask, label_id, config=config)
        if compute_network_context:
            row.update(network.get(int(label_id), {}))
        rows.append(row)
        if i == 1 or i % 25 == 0 or i == n_labels:
            print(f"  processed {i}/{n_labels} labels...")

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def analyze_labels_stack(
    labels_stack: np.ndarray,
    config: Optional[MitoMorphometricsConfig] = None,
    compute_network_context: bool = True,
) -> pd.DataFrame:
    """Analyze a stack of 2D label images (Z, Y, X) or (N, Y, X).

    Runs :func:`analyze_labels` per slice and adds a ``slice`` column to the
    combined DataFrame.
    """
    config = config or MitoMorphometricsConfig()
    labels_stack = np.asarray(labels_stack)
    rows = []
    n_slices = labels_stack.shape[0]
    for slice_idx in range(n_slices):
        print(f"Analyzing slice {slice_idx + 1}/{n_slices}...")
        df = analyze_labels(
            labels_stack[slice_idx],
            config=config,
            compute_network_context=compute_network_context,
        )
        if df.empty:
            continue
        df.insert(0, "slice", slice_idx)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
