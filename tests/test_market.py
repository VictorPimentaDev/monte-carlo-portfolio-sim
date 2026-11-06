import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from portfolio_sim.market import calibrate, load_prices


def test_calibration_corrects_log_drift_and_annualizes_sample_variance():
    logs = np.tile([-0.02, 0.01, 0.025, -0.005], 12)
    prices = pd.Series(100 * np.exp(np.r_[0, np.cumsum(logs)]))
    result = calibrate(prices, {"source": "test"})
    variance = logs.var(ddof=1) * 252
    assert result.volatility == pytest.approx(np.sqrt(variance))
    assert result.annual_return == pytest.approx(logs.mean() * 252 + variance / 2)
    assert result.provenance["observations"] == 48


@pytest.mark.parametrize("ticker", ["WEGE3", "^BVSP", None])
def test_offline_fixture_is_repeatable_and_never_claims_to_be_market_data(monkeypatch, ticker):
    monkeypatch.setitem(sys.modules, "yfinance", None)
    first = load_prices(ticker, offline=True)
    second = load_prices(ticker, offline=True)
    assert first == second
    assert first.provenance["kind"] == "synthetic_fixture"
    assert "NOT WEGE3 or IBOV" in first.provenance["source"]
    assert first.provenance["requested_ticker"] == ticker
    assert first.provenance["observations"] == 756
    assert first.provenance["first_price_date"] == "2022-01-03"
    assert first.provenance["last_price_date"] == "2024-11-26"
    assert first.annual_return == pytest.approx(0.08664353968956609, abs=1e-12)
    assert first.volatility == pytest.approx(0.25040837685909034, abs=1e-12)


@pytest.mark.parametrize(
    "symbol, expected",
    [("WEGE3", "WEGE3.SA"), (" wege3.sa ", "WEGE3.SA"), ("^BVSP", "^BVSP")],
)
def test_live_adapter_passes_adjustment_options_without_network(monkeypatch, symbol, expected):
    def download(ticker, **kwargs):
        assert ticker == expected
        assert kwargs["auto_adjust"] is True
        assert kwargs["multi_level_index"] is False
        assert kwargs["interval"] == "1d"
        assert kwargs["period"] == "5y"
        return pd.DataFrame(
            {"Close": np.exp(np.arange(50) / 100)},
            index=pd.bdate_range("2025-01-02", periods=50),
        )

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=download))
    result = load_prices(symbol)
    assert result.provenance["kind"] == "live_adjusted_prices"
    assert result.provenance["ticker"] == expected
    assert "retrieved_at_utc" in result.provenance
    assert result.annual_return == pytest.approx(2.52)
    assert result.volatility == pytest.approx(0, abs=1e-14)
    assert result.provenance["observations"] == 49
    assert result.provenance["first_price_date"] == "2025-01-02 00:00:00"


@pytest.mark.parametrize("frame", [None, pd.DataFrame(), pd.DataFrame({"Open": [100]})])
def test_empty_download_does_not_silently_substitute_synthetic_data(monkeypatch, frame):
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=lambda *a, **k: frame))
    with pytest.raises(ValueError, match="no daily prices"):
        load_prices("WEGE3")


def test_failed_download_does_not_silently_substitute_synthetic_data(monkeypatch):
    def download(*args, **kwargs):
        raise OSError("unavailable")

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=download))
    with pytest.raises(ValueError, match="price download failed"):
        load_prices("^BVSP")


def test_missing_optional_dependency_has_offline_guidance(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", None)
    with pytest.raises(ValueError, match="or use --offline"):
        load_prices("WEGE3")


@pytest.mark.parametrize("ticker", [None, "", "  "])
def test_live_calibration_requires_a_ticker(ticker):
    with pytest.raises(ValueError, match="ticker is required"):
        load_prices(ticker)


def test_constant_prices_have_zero_drift_and_volatility():
    result = calibrate(pd.Series([100.0] * 31), {})
    assert result.annual_return == 0
    assert result.volatility == 0
    assert result.provenance["observations"] == 30


@pytest.mark.parametrize(
    "prices",
    [
        pd.Series([100.0] * 30),
        pd.Series([100.0] * 30 + [0.0]),
        pd.Series([100.0] * 30 + [float("nan")]),
        pd.Series([100.0] * 31, index=[0] * 31),
        pd.Series([100.0] * 31, index=list(range(31))[::-1]),
    ],
)
def test_bad_price_series_is_rejected(prices):
    with pytest.raises(ValueError):
        calibrate(prices, {})
