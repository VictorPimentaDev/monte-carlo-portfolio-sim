from dataclasses import replace

import numpy as np
import pytest

from portfolio_sim.model import Config, analytical_statistics, simulate, summarize


def test_seed_reproduces_paths_and_changes_with_new_seed():
    config = Config(years=1.1, paths=8, seed=9)
    result = simulate(config)
    np.testing.assert_array_equal(result.values, simulate(config).values)
    assert not np.array_equal(result.values, simulate(replace(config, seed=10)).values)
    assert result.values.shape == (8, 15)
    assert result.times[-1] == 1.1
    assert (result.values[:, 0] == config.start).all()
    assert np.isfinite(result.values).all() and (result.values > 0).all()


def test_zero_volatility_is_exact_exponential_growth():
    config = Config(annual_return=-0.1, volatility=0, years=2.25, paths=5)
    result = simulate(config)
    expected = config.start * np.exp(config.annual_return * result.times)
    np.testing.assert_allclose(result.values, np.broadcast_to(expected, result.values.shape))
    assert result.statistics["loss_probability"] == 1
    for key, value in analytical_statistics(config).items():
        assert result.statistics[key] == pytest.approx(value)


def test_summary_uses_empirical_percentiles_and_strict_loss():
    values = np.array([50, 75, 100, 125, 200])
    assert summarize(values, 100) == {
        "p95": 185.0,
        "median": 100.0,
        "p5": 55.0,
        "mean": 110.0,
        "loss_probability": 0.4,
    }


def test_large_sample_matches_terminal_lognormal_distribution():
    result = simulate(Config(paths=50_000, seed=817))
    expected = analytical_statistics(result.config)
    for key in ("p5", "median", "p95", "mean"):
        assert result.statistics[key] == pytest.approx(expected[key], rel=0.025)
    assert result.statistics["loss_probability"] == pytest.approx(
        expected["loss_probability"], abs=0.006
    )
    log_terminal = np.log(result.values[:, -1] / result.config.start)
    assert log_terminal.var() == pytest.approx(0.17**2 * 10, rel=0.025)


@pytest.mark.parametrize(
    "change",
    [
        {"start": 0},
        {"start": -10},
        {"years": 0},
        {"years": float("inf")},
        {"annual_return": float("nan")},
        {"volatility": -0.01},
        {"paths": 0},
        {"paths": 1.5},
        {"paths": True},
        {"seed": -1},
        {"seed": True},
        {"paths": 20_000_001},
        {"years": 1001},
    ],
)
def test_invalid_parameters_fail_before_allocation(change):
    with pytest.raises(ValueError):
        Config(**change)


def test_overflow_is_an_error_not_infinite_paths():
    with pytest.raises(ValueError, match="numerical range"):
        simulate(Config(annual_return=1000, volatility=0, paths=1))


@pytest.mark.parametrize("name", ["start", "years", "annual_return", "volatility"])
@pytest.mark.parametrize("value", [True, "0.08", None, complex(1, 1)])
def test_numeric_parameters_reject_booleans_and_nonnumbers(name, value):
    with pytest.raises(ValueError, match="must be a real number"):
        Config(**{name: value})


@pytest.mark.parametrize("values", [[], [float("nan")], [0], [-1], [[1, 2]]])
def test_invalid_sample_is_rejected(values):
    with pytest.raises(ValueError):
        summarize(values, 100)
