"""Numerical kernels for Green-Kubo zero-shear viscosity."""

from __future__ import annotations

import numpy as np


GAS_CONSTANT_KJ_PER_MOL_K = 0.00831446261815324
KJ_PER_MOL_NM3_PS_TO_PA_S = 1.6605390671738467e-6


def autocorrelation_fft(values):
    """Return the unbiased autocorrelation of a one-dimensional series."""

    series = np.asarray(values, dtype=float)
    if series.ndim != 1 or len(series) < 2:
        raise ValueError("Autocorrelation requires a one-dimensional series.")
    if not np.all(np.isfinite(series)):
        raise ValueError("Autocorrelation input contains non-finite values.")
    centered = series - np.mean(series)
    fft_length = 1 << (2 * len(series) - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=fft_length)
    correlation = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_length)
    counts = np.arange(len(series), 0, -1, dtype=float)
    return correlation[: len(series)] / counts


def shear_relaxation_modulus(
    pressure_components,
    volume_nm3,
    temperature_k,
    max_lag_steps=None,
):
    """Compute the isotropically averaged shear relaxation modulus."""

    pressure = np.asarray(pressure_components, dtype=float)
    if pressure.ndim != 2 or pressure.shape[1] != 6:
        raise ValueError("Pressure data must have columns Pxx Pyy Pzz Pxy Pxz Pyz.")
    if len(pressure) < 2 or not np.all(np.isfinite(pressure)):
        raise ValueError("Pressure data must contain at least two finite samples.")
    if volume_nm3 <= 0.0 or temperature_k <= 0.0:
        raise ValueError("Volume and temperature must be positive.")

    if max_lag_steps is None:
        max_lag_steps = len(pressure) - 1
    max_lag_steps = int(max_lag_steps)
    if not 1 <= max_lag_steps < len(pressure):
        raise ValueError("Maximum lag is incompatible with the pressure series.")

    pxx, pyy, pzz, pxy, pxz, pyz = pressure.T
    off_diagonal = (
        autocorrelation_fft(pxy) + autocorrelation_fft(pxz) + autocorrelation_fft(pyz)
    )
    normal_differences = (
        autocorrelation_fft(pxx - pyy)
        + autocorrelation_fft(pxx - pzz)
        + autocorrelation_fft(pyy - pzz)
    )
    beta_volume = volume_nm3 / (GAS_CONSTANT_KJ_PER_MOL_K * temperature_k)
    modulus = beta_volume * (off_diagonal / 5.0 + normal_differences / 30.0)
    return modulus[: max_lag_steps + 1]


def cumulative_trapezoid(values, spacing):
    """Cumulative trapezoidal integral beginning at zero."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("Cumulative integration requires at least two values.")
    if spacing <= 0.0:
        raise ValueError("Integration spacing must be positive.")
    result = np.zeros_like(values)
    increments = 0.5 * (values[:-1] + values[1:]) * spacing
    result[1:] = np.cumsum(increments)
    return result


def viscosity_curve(modulus, sample_interval_ps):
    """Integrate a modulus curve and return viscosity in Pa s."""

    return cumulative_trapezoid(modulus, sample_interval_ps) * KJ_PER_MOL_NM3_PS_TO_PA_S


def plateau_average(lag_times_ps, viscosity_pa_s, start_ps, end_ps):
    """Average a running viscosity over an inclusive plateau window."""

    lag_times = np.asarray(lag_times_ps, dtype=float)
    viscosity = np.asarray(viscosity_pa_s, dtype=float)
    selected = (lag_times >= start_ps) & (lag_times <= end_ps)
    if np.count_nonzero(selected) < 2:
        raise ValueError("The plateau window must contain at least two lag samples.")
    return float(np.mean(viscosity[selected]))
