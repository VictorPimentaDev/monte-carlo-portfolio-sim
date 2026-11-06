"""Exact GBM transitions; return means annual arithmetic drift, not CAGR."""

from dataclasses import dataclass
from math import ceil, erf, exp, isfinite, log, sqrt
from statistics import NormalDist

import numpy as np


@dataclass(frozen=True)
class Config:
    start: float = 100_000
    years: float = 10
    annual_return: float = 0.08
    volatility: float = 0.17
    paths: int = 320
    seed: int = 42

    def __post_init__(self):
        for name in ("start", "years", "annual_return", "volatility"):
            if not isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.start <= 0 or self.years <= 0 or self.volatility < 0:
            raise ValueError("start/years must be positive and volatility nonnegative")
        if not isinstance(self.paths, int) or isinstance(self.paths, bool) or self.paths < 1:
            raise ValueError("paths must be a positive integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.years > 1_000 or self.paths * (ceil(self.years * 12) + 1) > 20_000_000:
            raise ValueError("request exceeds the 20 million sample / 1000 year limit")


@dataclass(frozen=True)
class Simulation:
    config: Config
    times: np.ndarray
    values: np.ndarray

    @property
    def statistics(self):
        return summarize(self.values[:, -1], self.config.start)


def summarize(values, start):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("values must be a finite, nonempty one-dimensional sample")
    if np.any(values <= 0) or not isfinite(start) or start <= 0:
        raise ValueError("values and start must be positive")
    p5, median, p95 = np.percentile(values, [5, 50, 95])
    return {
        "p95": float(p95),
        "median": float(median),
        "p5": float(p5),
        "mean": float(np.mean(values)),
        "loss_probability": float(np.mean(values < start)),
    }


def simulate(config: Config = Config()) -> Simulation:
    steps = ceil(config.years * 12)
    times = np.linspace(0, config.years, steps + 1)
    dt = config.years / steps
    shocks = np.random.default_rng(config.seed).standard_normal((config.paths, steps))
    try:
        with np.errstate(over="raise", under="raise", invalid="raise"):
            increments = (
                config.annual_return - 0.5 * config.volatility**2
            ) * dt + config.volatility * sqrt(dt) * shocks
            log_values = np.cumsum(increments, axis=1) + log(config.start)
            values = np.column_stack((np.full(config.paths, config.start), np.exp(log_values)))
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError("parameters exceed the numerical range of GBM") from exc
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("simulation must remain finite and strictly positive")
    return Simulation(config, times, values)


def analytical_statistics(config: Config):
    """Terminal lognormal distribution, useful for checking Monte Carlo error."""
    location = (config.annual_return - 0.5 * config.volatility**2) * config.years
    scale = config.volatility * sqrt(config.years)
    normal = NormalDist()
    try:
        result = {
            "p95": config.start * exp(location + scale * normal.inv_cdf(0.95)),
            "median": config.start * exp(location),
            "p5": config.start * exp(location + scale * normal.inv_cdf(0.05)),
            "mean": config.start * exp(config.annual_return * config.years),
            "loss_probability": (
                0.5 * (1 + erf(-location / scale / sqrt(2))) if scale else float(location < 0)
            ),
        }
    except OverflowError as exc:
        raise ValueError("parameters exceed the numerical range of GBM") from exc
    if any(not isfinite(value) for value in result.values()):
        raise ValueError("analytical statistics exceed the numerical range of GBM")
    return result
