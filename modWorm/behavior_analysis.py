"""Observational locomotion and population metrics.

These measurements do not feed back into the simulated animals. The axial sign
convention assumes centerline coordinate 0 is anterior and the final coordinate
is posterior; callers can reverse that convention for other data sources.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError


def _validate_time_series(time: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.asarray(time, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if time.ndim != 1 or len(time) < 2:
        raise ValueError("time must be a one-dimensional array with at least two samples")
    if x.ndim != 2 or y.shape != x.shape or x.shape[0] != len(time) or x.shape[1] < 2:
        raise ValueError("x and y must have shape (time samples, at least two body points)")
    if not (np.isfinite(time).all() and np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("time and body coordinates must be finite")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("time must be strictly increasing")
    return time, x, y


def _convex_hull_area(points: np.ndarray) -> float:
    if len(np.unique(points, axis=0)) < 3:
        return 0.0
    try:
        return float(ConvexHull(points).volume)
    except QhullError:
        return 0.0


def _runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    changes = np.flatnonzero(np.diff(labels) != 0) + 1
    starts = np.r_[0, changes]
    stops = np.r_[changes, len(labels)]
    return [(int(labels[start]), int(start), int(stop)) for start, stop in zip(starts, stops)]


def locomotion_metrics(
    time: np.ndarray,
    x_centerline: np.ndarray,
    y_centerline: np.ndarray,
    *,
    speed_threshold: float = 0.0,
    turn_rate_threshold: float = 0.5,
    anterior_at_index_zero: bool = True,
) -> dict[str, object]:
    """Measure signed movement, turns, curvature, bouts, and explored area.

    Thresholds define observational event segmentation only. They do not
    classify agar roaming/dwelling or alter neural or body dynamics.
    """
    time, x, y = _validate_time_series(time, x_centerline, y_centerline)
    if not np.isfinite(speed_threshold) or speed_threshold < 0.0:
        raise ValueError("speed_threshold must be finite and nonnegative")
    if not np.isfinite(turn_rate_threshold) or turn_rate_threshold < 0.0:
        raise ValueError("turn_rate_threshold must be finite and nonnegative")

    dt = np.diff(time)
    center = np.column_stack((x.mean(axis=1), y.mean(axis=1)))
    displacement = np.diff(center, axis=0)
    center_velocity = displacement / dt[:, None]
    orientation = np.column_stack((x[:, 0] - x[:, -1], y[:, 0] - y[:, -1]))
    if not anterior_at_index_zero:
        orientation = -orientation
    orientation_norm = np.linalg.norm(orientation, axis=1)
    if np.any(orientation_norm == 0.0):
        raise ValueError("head and tail coordinates must not coincide")
    orientation /= orientation_norm[:, None]
    interval_orientation = orientation[:-1] + orientation[1:]
    interval_norm = np.linalg.norm(interval_orientation, axis=1)
    interval_orientation[interval_norm > 0.0] /= interval_norm[interval_norm > 0.0, None]
    interval_orientation[interval_norm == 0.0] = orientation[:-1][interval_norm == 0.0]
    signed_velocity = np.einsum("ij,ij->i", center_velocity, interval_orientation)

    head_tail_angle = np.unwrap(np.arctan2(orientation[:, 1], orientation[:, 0]))
    angular_velocity = np.diff(head_tail_angle) / dt
    centerline = np.stack((x, y), axis=-1)
    first_edges = centerline[:, 1:-1] - centerline[:, :-2]
    second_edges = centerline[:, 2:] - centerline[:, 1:-1]
    cross = (
        first_edges[..., 0] * second_edges[..., 1]
        - first_edges[..., 1] * second_edges[..., 0]
    )
    denominator = (
        np.linalg.norm(first_edges, axis=-1)
        * np.linalg.norm(second_edges, axis=-1)
        * np.linalg.norm(first_edges + second_edges, axis=-1)
    )
    curvature = 2.0 * np.abs(cross) / np.maximum(denominator, np.finfo(float).eps)

    direction = np.sign(signed_velocity).astype(np.int8)
    direction[np.abs(signed_velocity) <= speed_threshold] = 0
    bouts = [
        {
            "direction": label,
            "start_time": float(time[start]),
            "end_time": float(time[min(stop, len(time) - 1)]),
            "duration": float(time[min(stop, len(time) - 1)] - time[start]),
        }
        for label, start, stop in _runs(direction)
    ]
    moving_directions = [bout["direction"] for bout in bouts if bout["direction"] != 0]
    reversal_count = sum(
        left != right
        for left, right in zip(moving_directions, moving_directions[1:])
    )
    turning = np.abs(angular_velocity) >= turn_rate_threshold
    turn_count = int(np.count_nonzero(turning & ~np.r_[False, turning[:-1]]))
    path_length = float(np.linalg.norm(displacement, axis=1).sum())

    return {
        "time": time,
        "center_of_mass": center,
        "signed_axial_velocity": signed_velocity,
        "interval_time": 0.5 * (time[:-1] + time[1:]),
        "angular_velocity": angular_velocity,
        "mean_absolute_curvature": float(np.mean(curvature)),
        "forward_bout_count": sum(bout["direction"] == 1 for bout in bouts),
        "backward_bout_count": sum(bout["direction"] == -1 for bout in bouts),
        "reversal_count": int(reversal_count),
        "turn_count": turn_count,
        "path_length": path_length,
        "net_displacement": float(np.linalg.norm(center[-1] - center[0])),
        "explored_convex_hull_area": _convex_hull_area(center),
        "bouts": bouts,
    }


def population_metrics(
    time: np.ndarray,
    positions: np.ndarray,
    *,
    cluster_distance: float,
    domain_area: float | None = None,
) -> dict[str, object]:
    """Measure proximity clusters and pairwise distances from worm centroids.

    Clusters are connected components of the pairwise-distance graph. This is
    a geometric grouping metric, not evidence of attraction or social intent.
    """
    time = np.asarray(time, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if time.ndim != 1 or len(time) < 1 or positions.ndim != 3:
        raise ValueError("time and positions must have shapes (frames,) and (frames, worms, 2)")
    if positions.shape != (len(time), positions.shape[1], 2) or positions.shape[1] < 1:
        raise ValueError("positions must have shape (time samples, at least one worm, 2)")
    if not np.isfinite(time).all() or not np.isfinite(positions).all():
        raise ValueError("time and positions must be finite")
    if len(time) > 1 and np.any(np.diff(time) <= 0.0):
        raise ValueError("time must be strictly increasing")
    if not np.isfinite(cluster_distance) or cluster_distance <= 0.0:
        raise ValueError("cluster_distance must be finite and greater than zero")
    if domain_area is not None and (not np.isfinite(domain_area) or domain_area <= 0.0):
        raise ValueError("domain_area must be finite and greater than zero")

    frame_distances = np.zeros((len(time), positions.shape[1], positions.shape[1]))
    cluster_counts = np.zeros(len(time), dtype=int)
    largest_clusters = np.zeros(len(time), dtype=int)
    encounters: list[dict[str, float | int]] = []
    pair_encounters: list[tuple[int, int, int]] = []
    active_start: dict[tuple[int, int], int] = {}

    for frame, points in enumerate(positions):
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        frame_distances[frame] = distances
        adjacency = distances <= cluster_distance
        np.fill_diagonal(adjacency, False)
        visited: set[int] = set()
        sizes: list[int] = []
        for root in range(len(points)):
            if root in visited:
                continue
            pending = [root]
            visited.add(root)
            size = 0
            while pending:
                current = pending.pop()
                size += 1
                for neighbor in np.flatnonzero(adjacency[current]):
                    neighbor = int(neighbor)
                    if neighbor not in visited:
                        visited.add(neighbor)
                        pending.append(neighbor)
            sizes.append(size)
        cluster_counts[frame] = sum(size > 1 for size in sizes)
        largest_clusters[frame] = max(sizes, default=0)

        pairs_now = {
            (i, j)
            for i in range(len(points))
            for j in range(i + 1, len(points))
            if distances[i, j] <= cluster_distance
        }
        for pair in pairs_now - active_start.keys():
            active_start[pair] = frame
        for pair in active_start.keys() - pairs_now:
            pair_encounters.append((*pair, active_start.pop(pair)))

    for pair, start in active_start.items():
        pair_encounters.append((*pair, start))
    for first, second, start in pair_encounters:
        end = next(
            (
                frame
                for frame in range(start + 1, len(time))
                if frame_distances[frame, first, second] > cluster_distance
            ),
            len(time) - 1,
        )
        encounters.append(
            {
                "worm_a": first,
                "worm_b": second,
                "start_time": float(time[start]),
                "end_time": float(time[end]),
                "duration": float(time[end] - time[start]),
            }
        )

    result: dict[str, object] = {
        "pairwise_distances": frame_distances,
        "cluster_count": cluster_counts,
        "largest_cluster_size": largest_clusters,
        "encounters": encounters,
    }
    if domain_area is not None:
        result["global_density_per_area"] = positions.shape[1] / domain_area
    return result
