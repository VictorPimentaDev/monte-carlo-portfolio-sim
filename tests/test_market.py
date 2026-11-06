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


def test_offline_fixture_is_repeatable_and_never_claims_to_be_wege():
    first = load_prices("WEGE3", offline=True)
    second = load_prices("WEGE3", offline=True)
    assert first == second
    assert first.provenance["kind"] == "synthetic_fixture"
    assert "NOT WEGE3 or IBOV" in first.provenance["source"]
    assert first.provenance["requested_ticker"] == "WEGE3"
    assert first.provenance["observations"] == 756


@pytest.mark.parametrize("symbol, expected", [("WEGE3", "WEGE3.SA"), ("^BVSP", "^BVSP")])
def test_live_adapter_passes_adjustment_options_without_network(monkeypatch, symbol, expected):
    def download(ticker, **kwargs):
        assert ticker == expected
        assert kwargs["auto_adjust"] is True
        assert kwargs["multi_level_index"] is False
        assert kwargs["interval"] == "1d"
        return pd.DataFrame({"Close": np.exp(np.arange(50) / 100)})

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=download))
    result = load_prices(symbol)
    assert result.provenance["kind"] == "live_adjusted_prices"
    assert result.provenance["ticker"] == expected
    assert "retrieved_at_utc" in result.provenance


def test_empty_download_does_not_silently_substitute_synthetic_data(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=lambda *a, **k: None))
    with pytest.raises(ValueError, match="no daily prices"):
        load_prices("WEGE3")


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
