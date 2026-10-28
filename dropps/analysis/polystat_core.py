"""NumPy-only kernels for many-molecule polymer statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MolecularShape:
    """Per-molecule size and shape values for one coordinate block."""

    end_to_end: np.ndarray
    radius_gyration: np.ndarray
    eigenvalues: np.ndarray
    relative_shape_anisotropy: np.ndarray


@dataclass(frozen=True)
class ScalarSummary:
    """Population statistics for one finite scalar sample."""

    count: int
    mean: float
    std: float
    rms: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class PersistenceResult:
    """First 1/e tangent-correlation crossing."""

    status: str
    steps: float
    contour_length: float
    lower_lag: int
    upper_lag: int


class RunningMoments:
    """Mergeable scalar moments using the parallel variance algorithm."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.sum_squares = 0.0
        self.minimum = np.inf
        self.maximum = -np.inf

    def update(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size == 0:
            return
        if not np.all(np.isfinite(values)):
            raise ValueError("Running statistics require finite values.")

        other_count = int(values.size)
        other_mean = float(np.mean(values))
        centered = values - other_mean
        other_m2 = float(np.dot(centered, centered))
        other_sum_squares = float(np.dot(values, values))
        other_minimum = float(np.min(values))
        other_maximum = float(np.max(values))
        self._merge_values(
            other_count,
            other_mean,
            other_m2,
            other_sum_squares,
            other_minimum,
            other_maximum,
        )

    def merge(self, other):
        if not isinstance(other, RunningMoments):
            raise TypeError("Can only merge another RunningMoments instance.")
        if other.count == 0:
            return
        self._merge_values(
            other.count,
            other.mean,
            other.m2,
            other.sum_squares,
            other.minimum,
            other.maximum,
        )

    def _merge_values(
        self,
        other_count,
        other_mean,
        other_m2,
        other_sum_squares,
        other_minimum,
        other_maximum,
    ):
        if self.count == 0:
            self.count = int(other_count)
            self.mean = float(other_mean)
            self.m2 = float(other_m2)
            self.sum_squares = float(other_sum_squares)
            self.minimum = float(other_minimum)
            self.maximum = float(other_maximum)
            return

        total = self.count + int(other_count)
        delta = float(other_mean) - self.mean
        self.m2 += float(other_m2) + delta * delta * self.count * other_count / total
        self.mean += delta * other_count / total
        self.sum_squares += float(other_sum_squares)
        self.count = total
        self.minimum = min(self.minimum, float(other_minimum))
        self.maximum = max(self.maximum, float(other_maximum))

    @property
    def variance(self):
        return self.m2 / self.count if self.count else np.nan

    @property
    def std(self):
        return np.sqrt(max(self.variance, 0.0)) if self.count else np.nan

    @property
    def rms(self):
        return np.sqrt(max(self.sum_squares / self.count, 0.0)) if self.count else np.nan

    def summary(self):
        if self.count == 0:
            return ScalarSummary(0, np.nan, np.nan, np.nan, np.nan, np.nan)
        return ScalarSummary(
            self.count,
            self.mean,
            float(self.std),
            float(self.rms),
            self.minimum,
            self.maximum,
        )


def _coordinate_block(coordinates, minimum_particles=1):
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 3 or coordinates.shape[2] != 3:
        raise ValueError("Coordinates must have shape (n_molecules, n_particles, 3).")
    if coordinates.shape[0] == 0 or coordinates.shape[1] < minimum_particles:
        raise ValueError(
            f"Coordinate blocks require at least one molecule and "
            f"{minimum_particles} particle(s) per molecule."
        )
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("Coordinates contain non-finite values.")
    return coordinates


def _weight_block(weights, n_molecules, n_particles):
    if weights is None:
        return np.ones((n_molecules, n_particles), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim == 1:
        if weights.shape != (n_particles,):
            raise ValueError("One-dimensional weights must match the particle count.")
        weights = np.broadcast_to(weights, (n_molecules, n_particles))
    elif weights.shape != (n_molecules, n_particles):
        raise ValueError(
            "Weights must have shape (n_particles,) or "
            "(n_molecules, n_particles)."
        )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Weights must be finite and non-negative.")
    if np.any(np.sum(weights, axis=1) <= 0.0):
        raise ValueError("Every molecule must have a positive total weight.")
    return weights


def summarize(values):
    """Return population statistics for a finite one-dimensional sample."""

    moments = RunningMoments()
    moments.update(values)
    return moments.summary()


def calculate_molecular_shape(coordinates, weights=None):
    """Calculate end-to-end distance, gyration eigenvalues, Rg, and kappa^2."""

    coordinates = _coordinate_block(coordinates)
    n_molecules, n_particles, _ = coordinates.shape
    weights = _weight_block(weights, n_molecules, n_particles)
    weight_sums = np.sum(weights, axis=1)
    centers = (
        np.einsum("mai,ma->mi", coordinates, weights)
        / weight_sums[:, np.newaxis]
    )
    centered = coordinates - centers[:, np.newaxis, :]
    tensors = np.einsum(
        "ma,mai,maj->mij",
        weights,
        centered,
        centered,
    ) / weight_sums[:, np.newaxis, np.newaxis]
    tensors = 0.5 * (tensors + np.swapaxes(tensors, 1, 2))

    eigenvalues = np.linalg.eigvalsh(tensors)[:, ::-1]
    scale = np.maximum(np.max(np.abs(eigenvalues), axis=1), 1.0)
    tolerance = 128.0 * np.finfo(np.float64).eps * scale
    if np.any(eigenvalues[:, -1] < -tolerance):
        minimum = float(np.min(eigenvalues[:, -1]))
        raise FloatingPointError(
            f"A gyration tensor has a negative eigenvalue ({minimum:.12g})."
        )
    eigenvalues = np.maximum(eigenvalues, 0.0)

    rg_squared = np.sum(eigenvalues, axis=1)
    radius_gyration = np.sqrt(rg_squared)
    mean_eigenvalue = rg_squared / 3.0
    numerator = 1.5 * np.sum(
        (eigenvalues - mean_eigenvalue[:, np.newaxis]) ** 2,
        axis=1,
    )
    relative_shape_anisotropy = np.full(n_molecules, np.nan, dtype=np.float64)
    nonzero = rg_squared > 0.0
    relative_shape_anisotropy[nonzero] = (
        numerator[nonzero] / rg_squared[nonzero] ** 2
    )
    relative_shape_anisotropy[nonzero] = np.clip(
        relative_shape_anisotropy[nonzero],
        0.0,
        1.0,
    )

    if n_particles >= 2:
        end_to_end = np.linalg.norm(
            coordinates[:, -1, :] - coordinates[:, 0, :],
            axis=1,
        )
    else:
        end_to_end = np.full(n_molecules, np.nan, dtype=np.float64)
    return MolecularShape(
        end_to_end=end_to_end,
        radius_gyration=radius_gyration,
        eigenvalues=eigenvalues,
        relative_shape_anisotropy=relative_shape_anisotropy,
    )


def end_to_end_distances(coordinates):
    """Return one first-to-last distance per molecule."""

    coordinates = _coordinate_block(coordinates, minimum_particles=2)
    return np.linalg.norm(coordinates[:, -1] - coordinates[:, 0], axis=1)


def _validate_lags(lags, maximum, include_zero=False):
    lags = np.asarray(lags, dtype=np.int64)
    if lags.ndim != 1 or lags.size == 0:
        raise ValueError("Lags must be a non-empty one-dimensional array.")
    minimum = 0 if include_zero else 1
    if np.any(lags < minimum) or np.any(lags > maximum):
        raise ValueError(f"Lags must be between {minimum} and {maximum}.")
    if np.unique(lags).size != lags.size or np.any(np.diff(lags) <= 0):
        raise ValueError("Lags must be unique and strictly increasing.")
    return lags


def _choose_lag_method(length, lag_count, method):
    if method not in {"auto", "direct", "fft"}:
        raise ValueError("Lag method must be 'auto', 'direct', or 'fft'.")
    if method != "auto":
        return method
    return "fft" if length >= 16 and lag_count >= max(8, length // 4) else "direct"


def _autocorrelation_sums(vectors):
    """Return summed non-circular dot autocorrelation for all nonnegative lags."""

    vector_count = vectors.shape[1]
    fft_length = 1 << (2 * vector_count - 1).bit_length()
    transformed = np.fft.rfft(vectors, n=fft_length, axis=1)
    power = np.sum(transformed * np.conjugate(transformed), axis=2)
    correlation = np.fft.irfft(power, n=fft_length, axis=1)[:, :vector_count]
    return correlation


def internal_distance_sums(coordinates, lags, method="auto"):
    """Return sums and counts of squared internal distances for requested lags."""

    coordinates = _coordinate_block(coordinates, minimum_particles=2)
    n_molecules, length, _ = coordinates.shape
    lags = _validate_lags(lags, length - 1)
    method = _choose_lag_method(length, lags.size, method)

    if method == "direct":
        sums = np.empty(lags.size, dtype=np.float64)
        for lag_index, lag in enumerate(lags):
            differences = coordinates[:, lag:, :] - coordinates[:, :-lag, :]
            sums[lag_index] = float(np.einsum("mai,mai->", differences, differences))
    else:
        centered = coordinates - np.mean(coordinates, axis=1, keepdims=True)
        autocorrelation = _autocorrelation_sums(centered)
        squared_norms = np.einsum("mai,mai->ma", centered, centered)
        prefix = np.concatenate(
            (
                np.zeros((n_molecules, 1), dtype=np.float64),
                np.cumsum(squared_norms, axis=1),
            ),
            axis=1,
        )
        sums = np.empty(lags.size, dtype=np.float64)
        for lag_index, lag in enumerate(lags):
            left = prefix[:, length - lag]
            right = prefix[:, length] - prefix[:, lag]
            per_molecule = left + right - 2.0 * autocorrelation[:, lag]
            scale = np.maximum(left + right, 1.0)
            tolerance = 512.0 * np.finfo(np.float64).eps * scale
            if np.any(per_molecule < -tolerance):
                raise FloatingPointError(
                    "FFT internal-distance calculation produced a negative sum."
                )
            sums[lag_index] = float(np.sum(np.maximum(per_molecule, 0.0)))

    counts = n_molecules * (length - lags)
    return sums, counts.astype(np.int64)


def _bond_lengths(coordinates):
    bonds = coordinates[:, 1:, :] - coordinates[:, :-1, :]
    lengths = np.linalg.norm(bonds, axis=2)
    if not np.all(np.isfinite(lengths)):
        raise ValueError("Backbone bond lengths contain non-finite values.")
    return bonds, lengths


def contour_distance_sums(coordinates, lags, tangent_midpoints=False):
    """Return summed contour distances for bead or tangent pairs."""

    coordinates = _coordinate_block(coordinates, minimum_particles=2)
    _, lengths = _bond_lengths(coordinates)
    prefix = np.concatenate(
        (
            np.zeros((coordinates.shape[0], 1), dtype=np.float64),
            np.cumsum(lengths, axis=1),
        ),
        axis=1,
    )

    if tangent_midpoints:
        maximum = lengths.shape[1] - 1
        lags = _validate_lags(lags, maximum, include_zero=True)
        midpoint_positions = prefix[:, :-1] + 0.5 * lengths
        sums = np.empty(lags.size, dtype=np.float64)
        counts = np.empty(lags.size, dtype=np.int64)
        for lag_index, lag in enumerate(lags):
            if lag == 0:
                sums[lag_index] = 0.0
                counts[lag_index] = midpoint_positions.shape[0] * midpoint_positions.shape[1]
            else:
                differences = (
                    midpoint_positions[:, lag:] - midpoint_positions[:, :-lag]
                )
                sums[lag_index] = float(np.sum(differences))
                counts[lag_index] = differences.size
        return sums, counts

    maximum = coordinates.shape[1] - 1
    lags = _validate_lags(lags, maximum)
    sums = np.empty(lags.size, dtype=np.float64)
    counts = np.empty(lags.size, dtype=np.int64)
    for lag_index, lag in enumerate(lags):
        differences = prefix[:, lag:] - prefix[:, :-lag]
        sums[lag_index] = float(np.sum(differences))
        counts[lag_index] = differences.size
    return sums, counts


def tangent_correlation_sums(coordinates, lags, method="auto"):
    """Return summed unit-tangent dot products and observation counts."""

    coordinates = _coordinate_block(coordinates, minimum_particles=3)
    bonds, lengths = _bond_lengths(coordinates)
    scale = np.maximum(np.max(lengths, axis=1), 1.0)
    tolerance = 128.0 * np.finfo(np.float64).eps * scale
    if np.any(lengths <= tolerance[:, np.newaxis]):
        raise ValueError("A selected backbone contains a zero-length bond.")
    tangents = bonds / lengths[:, :, np.newaxis]
    tangent_count = tangents.shape[1]
    lags = _validate_lags(lags, tangent_count - 1, include_zero=True)
    method = _choose_lag_method(tangent_count, lags.size, method)

    if method == "direct":
        sums = np.empty(lags.size, dtype=np.float64)
        for lag_index, lag in enumerate(lags):
            if lag == 0:
                sums[lag_index] = float(tangents.shape[0] * tangent_count)
            else:
                sums[lag_index] = float(
                    np.einsum(
                        "mai,mai->",
                        tangents[:, :-lag, :],
                        tangents[:, lag:, :],
                    )
                )
    else:
        autocorrelation = _autocorrelation_sums(tangents)
        sums = np.sum(autocorrelation[:, lags], axis=0)

    counts = tangents.shape[0] * (tangent_count - lags)
    sums = np.clip(sums, -counts, counts)
    return np.asarray(sums, dtype=np.float64), counts.astype(np.int64)


def persistence_1e(correlations, lags, contour_distances=None):
    """Find the first 1/e crossing using log-linear interpolation."""

    correlations = np.asarray(correlations, dtype=np.float64)
    lags = np.asarray(lags, dtype=np.int64)
    if correlations.ndim != 1 or correlations.shape != lags.shape:
        raise ValueError("Correlations and lags must be matching 1D arrays.")
    _validate_lags(lags, int(lags[-1]), include_zero=True)
    if lags[0] != 0:
        raise ValueError("Persistence calculation requires lag zero.")
    if not np.all(np.isfinite(correlations)):
        raise ValueError("Correlations must be finite.")
    if not np.isclose(correlations[0], 1.0, atol=1.0e-10, rtol=1.0e-10):
        raise ValueError("The lag-zero tangent correlation must equal one.")

    if contour_distances is None:
        contour_distances = lags.astype(np.float64)
    else:
        contour_distances = np.asarray(contour_distances, dtype=np.float64)
        if contour_distances.shape != lags.shape:
            raise ValueError("Contour distances must match the lag array.")
        if not np.all(np.isfinite(contour_distances)):
            raise ValueError("Contour distances must be finite.")
        if np.any(np.diff(contour_distances) < 0.0):
            raise ValueError("Contour distances must be non-decreasing.")

    threshold = np.exp(-1.0)
    for upper_index in range(1, lags.size):
        if correlations[upper_index] > threshold:
            continue
        lower_index = upper_index - 1
        lower_value = correlations[lower_index]
        upper_value = correlations[upper_index]
        if lower_value <= 0.0 or upper_value <= 0.0:
            return PersistenceResult(
                "nonpositive-bracket",
                np.nan,
                np.nan,
                int(lags[lower_index]),
                int(lags[upper_index]),
            )
        denominator = np.log(upper_value) - np.log(lower_value)
        if denominator == 0.0:
            return PersistenceResult(
                "flat-bracket",
                np.nan,
                np.nan,
                int(lags[lower_index]),
                int(lags[upper_index]),
            )
        fraction = (-1.0 - np.log(lower_value)) / denominator
        fraction = float(np.clip(fraction, 0.0, 1.0))
        steps = lags[lower_index] + fraction * (
            lags[upper_index] - lags[lower_index]
        )
        contour = contour_distances[lower_index] + fraction * (
            contour_distances[upper_index] - contour_distances[lower_index]
        )
        return PersistenceResult(
            "valid",
            float(steps),
            float(contour),
            int(lags[lower_index]),
            int(lags[upper_index]),
        )

    return PersistenceResult(
        "censored",
        np.nan,
        np.nan,
        int(lags[-1]),
        int(lags[-1]),
    )
