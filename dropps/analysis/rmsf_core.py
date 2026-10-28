"""Numerically stable streaming kernels for positional RMSF analysis."""

from __future__ import annotations

import numpy as np


class CoordinateMoments:
    """Accumulate per-site Cartesian moments without retaining trajectory frames.

    The second central moment uses Welford's online update.  Dividing it by
    ``count`` therefore gives the population covariance, matching the
    trajectory-average convention used by ``gmx rmsf``.
    """

    def __init__(self, site_count, reference=None):
        if int(site_count) != site_count or site_count <= 0:
            raise ValueError("RMSF site count must be a positive integer.")
        self.site_count = int(site_count)
        self.count = 0
        self.mean = np.zeros((self.site_count, 3), dtype=np.float64)
        self._second_moment = np.zeros(
            (self.site_count, 3, 3),
            dtype=np.float64,
        )

        if reference is None:
            self.reference = None
            self._reference_squared_displacement = None
        else:
            reference = _coordinates(reference, self.site_count, "Reference")
            self.reference = reference.copy()
            self._reference_squared_displacement = np.zeros(
                self.site_count,
                dtype=np.float64,
            )

    def update(self, coordinates):
        """Add one aligned coordinate frame."""

        coordinates = _coordinates(coordinates, self.site_count, "RMSF")
        self.count += 1
        delta = coordinates - self.mean
        self.mean += delta / self.count
        delta_after_mean_update = coordinates - self.mean
        self._second_moment += np.einsum(
            "ai,aj->aij",
            delta,
            delta_after_mean_update,
        )

        if self.reference is not None:
            displacement = coordinates - self.reference
            self._reference_squared_displacement += np.einsum(
                "ai,ai->a",
                displacement,
                displacement,
            )

    @property
    def covariance(self):
        """Return one population covariance tensor per site."""

        self._require_samples()
        covariance = self._second_moment / self.count
        # Welford is symmetric analytically; remove round-off antisymmetry so
        # eigendecompositions and serialized U tensors are deterministic.
        return 0.5 * (covariance + np.swapaxes(covariance, 1, 2))

    @property
    def rmsf(self):
        """Return ``sqrt(trace(covariance))`` for every site."""

        return rmsf_from_covariance(self.covariance)

    @property
    def reference_rmsd(self):
        """Return time-averaged positional RMSD from the fixed reference."""

        self._require_samples()
        if self._reference_squared_displacement is None:
            raise ValueError("Reference deviations were not enabled.")
        mean_square = self._reference_squared_displacement / self.count
        return np.sqrt(np.maximum(mean_square, 0.0))

    @property
    def reference_mean_square_displacement(self):
        """Return mean squared distance from the fixed reference per site."""

        self._require_samples()
        if self._reference_squared_displacement is None:
            raise ValueError("Reference deviations were not enabled.")
        return self._reference_squared_displacement / self.count

    def _require_samples(self):
        if self.count == 0:
            raise ValueError("At least one coordinate frame is required.")


def _coordinates(coordinates, site_count, label):
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.shape != (site_count, 3):
        raise ValueError(f"{label} coordinates must have shape ({site_count}, 3).")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError(f"{label} coordinates contain non-finite values.")
    return coordinates


def _covariances(covariances):
    covariances = np.asarray(covariances, dtype=np.float64)
    if covariances.ndim != 3 or covariances.shape[1:] != (3, 3):
        raise ValueError("Covariances must have shape (n_sites, 3, 3).")
    if covariances.shape[0] == 0:
        raise ValueError("At least one covariance tensor is required.")
    if not np.all(np.isfinite(covariances)):
        raise ValueError("Covariances contain non-finite values.")
    return covariances


def rmsf_from_covariance(covariances):
    """Calculate RMSF from Cartesian population covariance tensors."""

    covariances = _covariances(covariances)
    traces = np.trace(covariances, axis1=1, axis2=2)
    tolerance = (
        np.finfo(np.float64).eps
        * np.maximum(
            np.max(np.abs(covariances), axis=(1, 2)),
            1.0,
        )
        * 32.0
    )
    if np.any(traces < -tolerance):
        raise ValueError("A covariance tensor has a negative total variance.")
    return np.sqrt(np.maximum(traces, 0.0))


def aggregate_site_values(values, groups, weights):
    """Return weighted group means for values whose first axis is the site."""

    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim == 0 or values.shape[0] == 0:
        raise ValueError("Site values must have a non-empty first axis.")
    if weights.shape != (values.shape[0],):
        raise ValueError("Site weights must match the value first axis.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Site values contain non-finite values.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Site weights must be non-negative and finite.")

    aggregated = []
    for group_id, group in enumerate(groups):
        indices = np.asarray(group, dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError(f"Site group {group_id} must be a non-empty vector.")
        if np.unique(indices).size != indices.size:
            raise ValueError(f"Site group {group_id} contains duplicate sites.")
        if np.min(indices) < 0 or np.max(indices) >= values.shape[0]:
            raise IndexError(f"Site group {group_id} contains an invalid site.")
        group_weights = weights[indices]
        total = float(group_weights.sum())
        if total <= 0.0:
            raise ValueError(f"Site group {group_id} has no positive weight.")
        aggregated.append(
            np.tensordot(group_weights, values[indices], axes=(0, 0)) / total
        )
    if not aggregated:
        raise ValueError("At least one site group is required.")
    return np.stack(aggregated)


def b_factors_from_covariance(covariances):
    """Convert positional covariance in nm^2 to isotropic PDB B factors in A^2."""

    mean_square_fluctuation = np.square(rmsf_from_covariance(covariances))
    return (800.0 * np.pi**2 / 3.0) * mean_square_fluctuation
