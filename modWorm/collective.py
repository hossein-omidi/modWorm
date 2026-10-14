"""Small-group geometry primitives for a future synchronized shared-world solver.

This module computes pairwise capsule-contact forces on body centerline nodes.
It does not integrate those forces through modWorm's body ODE; a world adapter
must map nodal forces to generalized body forces before contact can change
locomotion.
"""

from __future__ import annotations

import numpy as np


def _closest_segment_points(
    a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Return segment interpolation fractions and their closest points."""
    first = a1 - a0
    second = b1 - b0
    offset = a0 - b0
    aa = float(np.dot(first, first))
    bb = float(np.dot(first, second))
    cc = float(np.dot(second, second))
    dd = float(np.dot(first, offset))
    ee = float(np.dot(second, offset))
    denominator = aa * cc - bb * bb
    epsilon = np.finfo(float).eps

    if aa <= epsilon and cc <= epsilon:
        first_fraction = second_fraction = 0.0
    elif aa <= epsilon:
        first_fraction = 0.0
        second_fraction = float(np.clip(ee / cc, 0.0, 1.0))
    elif cc <= epsilon:
        second_fraction = 0.0
        first_fraction = float(np.clip(-dd / aa, 0.0, 1.0))
    else:
        first_fraction = (
            float(np.clip((bb * ee - cc * dd) / denominator, 0.0, 1.0))
            if denominator > epsilon * aa * cc
            else 0.0
        )
        second_fraction = (bb * first_fraction + ee) / cc
        if second_fraction < 0.0:
            second_fraction = 0.0
            first_fraction = float(np.clip(-dd / aa, 0.0, 1.0))
        elif second_fraction > 1.0:
            second_fraction = 1.0
            first_fraction = float(np.clip((bb - dd) / aa, 0.0, 1.0))

    return (
        first_fraction,
        second_fraction,
        a0 + first_fraction * first,
        b0 + second_fraction * second,
    )


def capsule_contact_forces(
    centerline_a: np.ndarray,
    centerline_b: np.ndarray,
    *,
    radius_a: float,
    radius_b: float,
    stiffness: float,
    damping: float = 0.0,
    velocity_a: np.ndarray | None = None,
    velocity_b: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute equal-and-opposite penalty forces on two segmented centerlines.

    Segment pairs are treated as capsules. ``stiffness`` has force/length
    units, ``damping`` force*time/length, and positions and velocities must
    use consistent units. Damping is active only while contact points approach.
    The returned nodal arrays require a separate mechanical-coupling adapter.
    """
    first = np.asarray(centerline_a, dtype=float)
    second = np.asarray(centerline_b, dtype=float)
    if first.ndim != 2 or first.shape[1] != 2 or len(first) < 2:
        raise ValueError("centerline_a must have shape (at least two points, 2)")
    if second.ndim != 2 or second.shape[1] != 2 or len(second) < 2:
        raise ValueError("centerline_b must have shape (at least two points, 2)")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("centerline coordinates must be finite")
    if not np.isfinite(radius_a) or radius_a < 0.0 or not np.isfinite(radius_b) or radius_b < 0.0:
        raise ValueError("capsule radii must be finite and nonnegative")
    if not np.isfinite(stiffness) or stiffness < 0.0:
        raise ValueError("stiffness must be finite and nonnegative")
    if not np.isfinite(damping) or damping < 0.0:
        raise ValueError("damping must be finite and nonnegative")

    if velocity_a is None:
        velocity_first = np.zeros_like(first)
    else:
        velocity_first = np.asarray(velocity_a, dtype=float)
    if velocity_b is None:
        velocity_second = np.zeros_like(second)
    else:
        velocity_second = np.asarray(velocity_b, dtype=float)
    if velocity_first.shape != first.shape or velocity_second.shape != second.shape:
        raise ValueError("velocity arrays must match their respective centerline shapes")
    if not np.isfinite(velocity_first).all() or not np.isfinite(velocity_second).all():
        raise ValueError("centerline velocities must be finite")

    forces_a = np.zeros_like(first)
    forces_b = np.zeros_like(second)
    combined_radius = radius_a + radius_b
    center_direction = first.mean(axis=0) - second.mean(axis=0)
    center_norm = float(np.linalg.norm(center_direction))

    for index_a in range(len(first) - 1):
        for index_b in range(len(second) - 1):
            weight_a, weight_b, point_a, point_b = _closest_segment_points(
                first[index_a], first[index_a + 1],
                second[index_b], second[index_b + 1],
            )
            delta = point_a - point_b
            distance = float(np.linalg.norm(delta))
            overlap = combined_radius - distance
            if overlap <= 0.0:
                continue
            if distance > np.finfo(float).eps:
                normal = delta / distance
            elif center_norm > np.finfo(float).eps:
                normal = center_direction / center_norm
            else:
                tangent = first[index_a + 1] - first[index_a]
                tangent_norm = float(np.linalg.norm(tangent))
                if tangent_norm <= np.finfo(float).eps:
                    raise ValueError("contact normal is undefined for coincident degenerate segments")
                normal = np.array([-tangent[1], tangent[0]]) / tangent_norm

            contact_velocity_a = (
                (1.0 - weight_a) * velocity_first[index_a]
                + weight_a * velocity_first[index_a + 1]
            )
            contact_velocity_b = (
                (1.0 - weight_b) * velocity_second[index_b]
                + weight_b * velocity_second[index_b + 1]
            )
            closing_speed = max(0.0, -float(np.dot(contact_velocity_a - contact_velocity_b, normal)))
            magnitude = stiffness * overlap + damping * closing_speed
            force = magnitude * normal
            forces_a[index_a] += (1.0 - weight_a) * force
            forces_a[index_a + 1] += weight_a * force
            forces_b[index_b] -= (1.0 - weight_b) * force
            forces_b[index_b + 1] -= weight_b * force

    return forces_a, forces_b
