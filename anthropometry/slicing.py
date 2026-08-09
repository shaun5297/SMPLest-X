#!/usr/bin/env python3
"""Body-part-agnostic horizontal slicing for closed triangular meshes.

The public entry point is :func:`slice_mesh`.  It intersects a triangular mesh
with ``y = plane_y`` and returns every reconstructable closed contour.  The
module deliberately contains no torso, waist, chest, or hip selection logic.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable

import numpy as np


EPS = 1e-8
ENDPOINT_CLUSTER_TOLERANCE_M = 1e-6

EndpointKey = tuple[str, int] | tuple[str, int, int]


def _validate_mesh(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (F, 3), got {faces.shape}")
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("vertices and faces must be non-empty")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("faces contain an out-of-range vertex index")
    if not np.isfinite(vertices).all():
        raise ValueError("vertices contain NaN or Inf")
    return vertices, faces


def _unique_points(points: Iterable[np.ndarray], tolerance: float) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for point in points:
        if not any(np.linalg.norm(point - existing) <= tolerance for existing in unique):
            unique.append(point)
    return unique


def _intersect_triangle(
    triangle: np.ndarray,
    plane_y: float,
    eps: float,
) -> tuple[np.ndarray | None, str]:
    """Intersect one triangle using fixed rules for on-plane degeneracies.

    Coplanar faces are ignored.  An on-plane edge of a non-coplanar triangle is
    emitted and later deduplicated as an undirected segment.  A single on-plane
    vertex is emitted only when the other two vertices lie on opposite sides;
    a tangential point contact does not form a segment.
    """
    distances = triangle[:, 1] - plane_y
    distances[np.abs(distances) <= eps] = 0.0
    on_plane = np.flatnonzero(distances == 0.0)

    if len(on_plane) == 3:
        return None, "coplanar_face"
    if len(on_plane) == 2:
        return triangle[on_plane].copy(), "on_plane_edge"
    if len(on_plane) == 1:
        vertex_index = int(on_plane[0])
        other = [index for index in range(3) if index != vertex_index]
        if distances[other[0]] * distances[other[1]] < 0.0:
            first, second = other
            t = distances[first] / (distances[first] - distances[second])
            crossing = triangle[first] + t * (triangle[second] - triangle[first])
            crossing[1] = plane_y
            return np.stack([triangle[vertex_index], crossing]), "vertex_crossing"
        return None, "point_contact"

    positive = distances > 0.0
    if positive.all() or (~positive).all():
        return None, "none"

    intersections = []
    for first, second in ((0, 1), (1, 2), (2, 0)):
        if distances[first] * distances[second] < 0.0:
            t = distances[first] / (distances[first] - distances[second])
            point = triangle[first] + t * (triangle[second] - triangle[first])
            point[1] = plane_y
            intersections.append(point)
    intersections = _unique_points(intersections, eps)
    if len(intersections) != 2:
        return None, "ambiguous"
    return np.stack(intersections), "crossing"


def intersect_mesh_with_horizontal_plane(
    vertices: np.ndarray,
    faces: np.ndarray,
    plane_y: float,
    *,
    eps: float = EPS,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Return raw triangle-plane intersection segments shaped ``(S, 2, 3)``."""
    vertices, faces = _validate_mesh(vertices, faces)
    if not np.isfinite(plane_y):
        raise ValueError("plane_y must be finite")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    triangles = vertices[faces]
    min_y = triangles[:, :, 1].min(axis=1)
    max_y = triangles[:, :, 1].max(axis=1)
    candidate_indices = np.flatnonzero(
        (min_y <= plane_y + eps) & (max_y >= plane_y - eps)
    )
    segments = []
    counts: defaultdict[str, int] = defaultdict(int)
    for triangle_index in candidate_indices:
        segment, case = _intersect_triangle(triangles[triangle_index], plane_y, eps)
        counts[case] += 1
        if segment is not None:
            segment[:, 1] = plane_y
            segments.append(segment)

    if segments:
        segment_array = np.asarray(segments, dtype=np.float64)
    else:
        segment_array = np.empty((0, 2, 3), dtype=np.float64)
    diagnostics: dict[str, int | float] = {
        "plane_y_m": float(plane_y),
        "mesh_face_count": int(len(faces)),
        "candidate_face_count": int(len(candidate_indices)),
        "raw_segment_count": int(len(segment_array)),
        "crossing_face_count": int(counts["crossing"]),
        "vertex_crossing_face_count": int(counts["vertex_crossing"]),
        "on_plane_edge_face_count": int(counts["on_plane_edge"]),
        "coplanar_face_count": int(counts["coplanar_face"]),
        "point_contact_face_count": int(counts["point_contact"]),
        "ambiguous_face_count": int(counts["ambiguous"]),
    }
    return segment_array, diagnostics


def _vertex_key(vertex_id: int) -> EndpointKey:
    return ("vertex", int(vertex_id))


def _edge_key(first_vertex_id: int, second_vertex_id: int) -> EndpointKey:
    first, second = sorted((int(first_vertex_id), int(second_vertex_id)))
    return ("edge", first, second)


def _intersect_triangle_with_provenance(
    triangle: np.ndarray,
    vertex_ids: np.ndarray,
    plane_y: float,
    eps: float,
) -> tuple[np.ndarray | None, tuple[EndpointKey, EndpointKey] | None, str]:
    """Intersect one triangle while retaining mesh-topological endpoint identity."""
    distances = triangle[:, 1] - plane_y
    distances[np.abs(distances) <= eps] = 0.0
    on_plane = np.flatnonzero(distances == 0.0)

    if len(on_plane) == 3:
        return None, None, "coplanar_face"
    if len(on_plane) == 2:
        first, second = (int(index) for index in on_plane)
        return (
            triangle[[first, second]].copy(),
            (_vertex_key(vertex_ids[first]), _vertex_key(vertex_ids[second])),
            "on_plane_edge",
        )
    if len(on_plane) == 1:
        vertex_index = int(on_plane[0])
        other = [index for index in range(3) if index != vertex_index]
        if distances[other[0]] * distances[other[1]] < 0.0:
            first, second = other
            t = distances[first] / (distances[first] - distances[second])
            crossing = triangle[first] + t * (triangle[second] - triangle[first])
            crossing[1] = plane_y
            return (
                np.stack([triangle[vertex_index], crossing]),
                (
                    _vertex_key(vertex_ids[vertex_index]),
                    _edge_key(vertex_ids[first], vertex_ids[second]),
                ),
                "vertex_crossing",
            )
        return None, None, "point_contact"

    positive = distances > 0.0
    if positive.all() or (~positive).all():
        return None, None, "none"

    intersections: list[np.ndarray] = []
    keys: list[EndpointKey] = []
    for first, second in ((0, 1), (1, 2), (2, 0)):
        if distances[first] * distances[second] < 0.0:
            t = distances[first] / (distances[first] - distances[second])
            point = triangle[first] + t * (triangle[second] - triangle[first])
            point[1] = plane_y
            intersections.append(point)
            keys.append(_edge_key(vertex_ids[first], vertex_ids[second]))
    if len(intersections) != 2 or len(set(keys)) != 2:
        return None, None, "ambiguous"
    return np.stack(intersections), (keys[0], keys[1]), "crossing"


def _intersect_mesh_with_horizontal_plane_provenance(
    vertices: np.ndarray,
    faces: np.ndarray,
    plane_y: float,
    *,
    eps: float = EPS,
) -> tuple[
    np.ndarray,
    list[tuple[EndpointKey, EndpointKey]],
    dict[str, int | float],
]:
    """Return raw segments together with vertex/edge endpoint provenance."""
    vertices, faces = _validate_mesh(vertices, faces)
    if not np.isfinite(plane_y):
        raise ValueError("plane_y must be finite")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    triangles = vertices[faces]
    min_y = triangles[:, :, 1].min(axis=1)
    max_y = triangles[:, :, 1].max(axis=1)
    candidate_indices = np.flatnonzero(
        (min_y <= plane_y + eps) & (max_y >= plane_y - eps)
    )
    segments: list[np.ndarray] = []
    endpoint_keys: list[tuple[EndpointKey, EndpointKey]] = []
    counts: defaultdict[str, int] = defaultdict(int)
    for triangle_index in candidate_indices:
        segment, keys, case = _intersect_triangle_with_provenance(
            triangles[triangle_index],
            faces[triangle_index],
            plane_y,
            eps,
        )
        counts[case] += 1
        if segment is not None and keys is not None:
            segment[:, 1] = plane_y
            segments.append(segment)
            endpoint_keys.append(keys)

    segment_array = (
        np.asarray(segments, dtype=np.float64)
        if segments
        else np.empty((0, 2, 3), dtype=np.float64)
    )
    diagnostics: dict[str, int | float] = {
        "plane_y_m": float(plane_y),
        "mesh_face_count": int(len(faces)),
        "candidate_face_count": int(len(candidate_indices)),
        "raw_segment_count": int(len(segment_array)),
        "crossing_face_count": int(counts["crossing"]),
        "vertex_crossing_face_count": int(counts["vertex_crossing"]),
        "on_plane_edge_face_count": int(counts["on_plane_edge"]),
        "coplanar_face_count": int(counts["coplanar_face"]),
        "point_contact_face_count": int(counts["point_contact"]),
        "ambiguous_face_count": int(counts["ambiguous"]),
    }
    return segment_array, endpoint_keys, diagnostics


def _connect_provenance_segments_to_loops(
    segments: np.ndarray,
    endpoint_keys: list[tuple[EndpointKey, EndpointKey]],
    *,
    diagnostic_tolerance: float,
) -> tuple[list[np.ndarray], dict[str, object]]:
    """Connect intersection segments using mesh topology, never XYZ proximity."""
    segments = np.asarray(segments, dtype=np.float64)
    if segments.ndim != 3 or segments.shape[1:] != (2, 3):
        raise ValueError(f"segments must have shape (S, 2, 3), got {segments.shape}")
    if len(endpoint_keys) != len(segments):
        raise ValueError("every segment must have one endpoint-key pair")
    if not np.isfinite(segments).all():
        raise ValueError("segments contain NaN or Inf")
    if diagnostic_tolerance <= 0.0:
        raise ValueError("diagnostic_tolerance must be positive")

    point_samples: defaultdict[EndpointKey, list[np.ndarray]] = defaultdict(list)
    unique_edges: dict[
        tuple[EndpointKey, EndpointKey], tuple[np.ndarray, np.ndarray]
    ] = {}
    topological_zero_count = 0
    duplicate_count = 0
    for segment, (first_key, second_key) in zip(segments, endpoint_keys):
        if first_key == second_key:
            topological_zero_count += 1
            continue
        point_samples[first_key].append(segment[0])
        point_samples[second_key].append(segment[1])
        edge = tuple(sorted((first_key, second_key)))
        if edge in unique_edges:
            duplicate_count += 1
        else:
            unique_edges[edge] = (segment[0], segment[1])

    points = {
        key: np.mean(np.asarray(samples, dtype=np.float64), axis=0)
        for key, samples in point_samples.items()
    }
    max_coordinate_disagreement = max(
        (
            float(np.linalg.norm(sample - points[key]))
            for key, samples in point_samples.items()
            for sample in samples
        ),
        default=0.0,
    )
    valid_lengths = [
        float(np.linalg.norm(points[first] - points[second]))
        for first, second in unique_edges
    ]
    very_short_count = sum(
        length <= diagnostic_tolerance for length in valid_lengths
    )

    adjacency: defaultdict[EndpointKey, set[EndpointKey]] = defaultdict(set)
    for first, second in unique_edges:
        adjacency[first].add(second)
        adjacency[second].add(first)

    loops = []
    component_diagnostics = []
    remaining = set(adjacency)
    while remaining:
        seed = min(remaining)
        queue = deque([seed])
        component = set()
        while queue:
            node = queue.popleft()
            if node in component:
                continue
            component.add(node)
            queue.extend(adjacency[node] - component)
        remaining -= component
        degrees = {node: len(adjacency[node]) for node in component}
        edge_count = sum(degrees.values()) // 2
        is_closed = bool(component) and all(degree == 2 for degree in degrees.values())
        component_record: dict[str, object] = {
            "node_count": int(len(component)),
            "edge_count": int(edge_count),
            "min_degree": int(min(degrees.values())) if degrees else 0,
            "max_degree": int(max(degrees.values())) if degrees else 0,
            "all_degrees_two": bool(is_closed),
        }
        if not is_closed:
            component_diagnostics.append(component_record)
            continue

        start = min(component)
        ordered_keys = [start]
        previous = None
        current = start
        for _ in range(len(component)):
            choices = sorted(
                adjacency[current]
                - ({previous} if previous is not None else set())
            )
            if not choices:
                break
            next_node = choices[0]
            if next_node == start:
                break
            if next_node in ordered_keys:
                break
            ordered_keys.append(next_node)
            previous, current = current, next_node
        traversal_closed = (
            len(ordered_keys) == len(component)
            and start in adjacency[ordered_keys[-1]]
        )
        component_record["traversal_closed"] = bool(traversal_closed)
        component_diagnostics.append(component_record)
        if traversal_closed:
            loops.append(
                np.asarray([points[key] for key in ordered_keys], dtype=np.float64)
            )

    diagnostics = {
        "input_segment_count": int(len(segments)),
        "zero_length_segments_removed": int(topological_zero_count),
        "topological_zero_length_segments_removed": int(topological_zero_count),
        "duplicate_segments_removed": int(duplicate_count),
        "unique_segment_count": int(len(unique_edges)),
        "remaining_duplicate_segments": 0,
        "endpoint_node_count": int(len(points)),
        "endpoint_identity": "mesh_topology_provenance",
        "coordinate_tolerance_controls_topology": False,
        "endpoint_cluster_tolerance_m": float(diagnostic_tolerance),
        "max_endpoint_cluster_error_m": float(max_coordinate_disagreement),
        "max_same_provenance_coordinate_disagreement_m": float(
            max_coordinate_disagreement
        ),
        "very_short_but_topologically_valid_segment_count": int(
            very_short_count
        ),
        "minimum_valid_segment_length_m": (
            float(min(valid_lengths)) if valid_lengths else None
        ),
        "connected_component_count": int(len(component_diagnostics)),
        "closed_loop_count": int(len(loops)),
        "invalid_component_count": int(
            sum(
                not item.get("traversal_closed", False)
                for item in component_diagnostics
            )
        ),
        "components": component_diagnostics,
    }
    return loops, diagnostics


def _point_key(point: np.ndarray, tolerance: float) -> tuple[int, int, int]:
    return tuple(np.rint(point / tolerance).astype(np.int64))


def deduplicate_segments(
    segments: np.ndarray,
    *,
    tolerance: float = ENDPOINT_CLUSTER_TOLERANCE_M,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Cluster endpoints and remove zero-length and duplicate undirected edges."""
    segments = np.asarray(segments, dtype=np.float64)
    if segments.ndim != 3 or segments.shape[1:] != (2, 3):
        raise ValueError(f"segments must have shape (S, 2, 3), got {segments.shape}")
    if not np.isfinite(segments).all():
        raise ValueError("segments contain NaN or Inf")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")

    point_sums: dict[tuple[int, int, int], np.ndarray] = {}
    point_counts: defaultdict[tuple[int, int, int], int] = defaultdict(int)
    keyed_segments = []
    zero_length_count = 0
    for segment in segments:
        first_key = _point_key(segment[0], tolerance)
        second_key = _point_key(segment[1], tolerance)
        if first_key == second_key or np.linalg.norm(segment[0] - segment[1]) <= tolerance:
            zero_length_count += 1
            continue
        for key, point in ((first_key, segment[0]), (second_key, segment[1])):
            if key not in point_sums:
                point_sums[key] = np.zeros(3, dtype=np.float64)
            point_sums[key] += point
            point_counts[key] += 1
        keyed_segments.append((first_key, second_key))

    point_means = {
        key: point_sums[key] / point_counts[key]
        for key in point_sums
    }
    unique_edges: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()
    duplicate_count = 0
    for first_key, second_key in keyed_segments:
        edge = tuple(sorted((first_key, second_key)))
        if edge in unique_edges:
            duplicate_count += 1
        else:
            unique_edges.add(edge)

    unique_segments = np.asarray(
        [[point_means[first], point_means[second]] for first, second in sorted(unique_edges)],
        dtype=np.float64,
    )
    if not len(unique_edges):
        unique_segments = np.empty((0, 2, 3), dtype=np.float64)

    max_cluster_radius = 0.0
    for segment in segments:
        for point in segment:
            key = _point_key(point, tolerance)
            if key in point_means:
                max_cluster_radius = max(
                    max_cluster_radius,
                    float(np.linalg.norm(point - point_means[key])),
                )
    diagnostics: dict[str, int | float] = {
        "input_segment_count": int(len(segments)),
        "zero_length_segments_removed": int(zero_length_count),
        "duplicate_segments_removed": int(duplicate_count),
        "unique_segment_count": int(len(unique_segments)),
        "remaining_duplicate_segments": 0,
        "endpoint_node_count": int(len(point_means)),
        "max_endpoint_cluster_error_m": float(max_cluster_radius),
        "endpoint_cluster_tolerance_m": float(tolerance),
    }
    return unique_segments, diagnostics


def connect_segments_to_loops(
    segments: np.ndarray,
    *,
    tolerance: float = ENDPOINT_CLUSTER_TOLERANCE_M,
) -> tuple[list[np.ndarray], dict[str, object]]:
    """Build an undirected endpoint graph and traverse every degree-2 component."""
    unique_segments, deduplication = deduplicate_segments(
        segments, tolerance=tolerance
    )
    point_sums: dict[tuple[int, int, int], np.ndarray] = {}
    point_counts: defaultdict[tuple[int, int, int], int] = defaultdict(int)
    adjacency: defaultdict[tuple[int, int, int], set[tuple[int, int, int]]] = defaultdict(set)
    for segment in unique_segments:
        first_key = _point_key(segment[0], tolerance)
        second_key = _point_key(segment[1], tolerance)
        for key, point in ((first_key, segment[0]), (second_key, segment[1])):
            if key not in point_sums:
                point_sums[key] = np.zeros(3, dtype=np.float64)
            point_sums[key] += point
            point_counts[key] += 1
        adjacency[first_key].add(second_key)
        adjacency[second_key].add(first_key)
    points = {key: point_sums[key] / point_counts[key] for key in point_sums}

    loops = []
    component_diagnostics = []
    remaining = set(adjacency)
    while remaining:
        seed = min(remaining)
        queue = deque([seed])
        component = set()
        while queue:
            node = queue.popleft()
            if node in component:
                continue
            component.add(node)
            queue.extend(adjacency[node] - component)
        remaining -= component
        degrees = {node: len(adjacency[node]) for node in component}
        edge_count = sum(degrees.values()) // 2
        is_closed = bool(component) and all(degree == 2 for degree in degrees.values())
        component_record: dict[str, object] = {
            "node_count": int(len(component)),
            "edge_count": int(edge_count),
            "min_degree": int(min(degrees.values())) if degrees else 0,
            "max_degree": int(max(degrees.values())) if degrees else 0,
            "all_degrees_two": bool(is_closed),
        }
        if not is_closed:
            component_diagnostics.append(component_record)
            continue

        start = min(component)
        ordered_keys = [start]
        previous = None
        current = start
        for _ in range(len(component)):
            choices = sorted(adjacency[current] - ({previous} if previous is not None else set()))
            if not choices:
                break
            next_node = choices[0]
            if next_node == start:
                break
            if next_node in ordered_keys:
                break
            ordered_keys.append(next_node)
            previous, current = current, next_node
        traversal_closed = len(ordered_keys) == len(component) and start in adjacency[ordered_keys[-1]]
        component_record["traversal_closed"] = bool(traversal_closed)
        component_diagnostics.append(component_record)
        if traversal_closed:
            loops.append(np.asarray([points[key] for key in ordered_keys], dtype=np.float64))

    diagnostics = {
        **deduplication,
        "connected_component_count": int(len(component_diagnostics)),
        "closed_loop_count": int(len(loops)),
        "invalid_component_count": int(
            sum(not item.get("traversal_closed", False) for item in component_diagnostics)
        ),
        "components": component_diagnostics,
    }
    return loops, diagnostics


def compute_contour_perimeter(points: np.ndarray) -> float:
    """Compute closed contour perimeter in the slicing plane (X-Z)."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("a contour must contain at least three 3D points")
    xz = points[:, [0, 2]]
    return float(np.linalg.norm(np.roll(xz, -1, axis=0) - xz, axis=1).sum())


def _signed_area_xz(points: np.ndarray) -> float:
    x = points[:, 0]
    z = points[:, 2]
    return float(0.5 * np.sum(x * np.roll(z, -1) - np.roll(x, -1) * z))


def compute_contour_area(points: np.ndarray) -> float:
    """Compute unsigned X-Z planar area with the shoelace formula."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("a contour must contain at least three 3D points")
    return abs(_signed_area_xz(points))


def compute_contour_centroid_xz(points: np.ndarray) -> np.ndarray:
    """Compute the X-Z polygon centroid with a mean fallback for tiny areas."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("a contour must contain at least three 3D points")
    x = points[:, 0]
    z = points[:, 2]
    cross = x * np.roll(z, -1) - np.roll(x, -1) * z
    signed_area = 0.5 * cross.sum()
    if abs(signed_area) <= EPS:
        return points[:, [0, 2]].mean(axis=0)
    centroid_x = np.sum((x + np.roll(x, -1)) * cross) / (6.0 * signed_area)
    centroid_z = np.sum((z + np.roll(z, -1)) * cross) / (6.0 * signed_area)
    return np.asarray([centroid_x, centroid_z], dtype=np.float64)


def slice_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    plane_y: float,
    *,
    eps: float = EPS,
    endpoint_tolerance: float = ENDPOINT_CLUSTER_TOLERANCE_M,
) -> dict[str, object]:
    """Slice a mesh and connect endpoints by mesh-topological provenance."""
    raw_segments, endpoint_keys, intersection = (
        _intersect_mesh_with_horizontal_plane_provenance(
        vertices, faces, plane_y, eps=eps
        )
    )
    loops, connectivity = _connect_provenance_segments_to_loops(
        raw_segments,
        endpoint_keys,
        diagnostic_tolerance=endpoint_tolerance,
    )
    contours = []
    for contour_id, points in enumerate(loops):
        if _signed_area_xz(points) < 0.0:
            points = points[::-1].copy()
        perimeter = compute_contour_perimeter(points)
        area = compute_contour_area(points)
        centroid = compute_contour_centroid_xz(points)
        contours.append(
            {
                "id": int(contour_id),
                "ordered_points_m": points.tolist(),
                "num_points": int(len(points)),
                "perimeter_m": float(perimeter),
                "perimeter_cm": float(perimeter * 100.0),
                "area_m2": float(area),
                "area_cm2": float(area * 10_000.0),
                "centroid_xz_m": centroid.tolist(),
                "plane_y_m": float(plane_y),
                "y_span_m": float(np.ptp(points[:, 1])),
                "closure_error_m": 0.0,
                "all_node_degrees_two": True,
            }
        )
    contours.sort(key=lambda item: item["area_m2"], reverse=True)
    for contour_id, contour in enumerate(contours):
        contour["id"] = contour_id
    return {
        "plane_y_m": float(plane_y),
        "contours": contours,
        "diagnostics": {
            "intersection": intersection,
            "connectivity": connectivity,
        },
    }
