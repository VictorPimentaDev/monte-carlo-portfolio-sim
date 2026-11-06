"""Optional adjusted daily prices. Offline data is explicitly synthetic."""

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.resources import files

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Calibration:
    annual_return: float
    volatility: float
    provenance: dict

    def to_dict(self):
        return asdict(self)


def calibrate(prices: pd.Series, provenance: dict) -> Calibration:
    """Use sample log-return volatility and correct log drift to GBM drift."""
    if not isinstance(prices, pd.Series) or len(prices) < 31:
        raise ValueError("at least 31 daily prices are required")
    if not prices.index.is_monotonic_increasing or not prices.index.is_unique:
        raise ValueError("price dates must be unique and increasing")
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("daily prices must be finite and positive")
    returns = np.diff(np.log(values))
    volatility = float(np.std(returns, ddof=1) * np.sqrt(252))
    annual_return = float(np.mean(returns) * 252 + 0.5 * volatility**2)
    return Calibration(
        annual_return,
        volatility,
        {
            **provenance,
            "observations": len(returns),
            "first_price_date": str(prices.index[0]),
            "last_price_date": str(prices.index[-1]),
            "annualization_days": 252,
            "method": "mean(log returns)*252 + 0.5*annualized_variance; sample std(ddof=1)",
        },
    )


def load_prices(ticker: str | None = None, *, offline: bool = False) -> Calibration:
    if offline:
        fixture = files("portfolio_sim").joinpath("fixtures")
        frame = pd.read_csv(fixture.joinpath("synthetic_daily.csv"), index_col="date")
        provenance = json.loads(fixture.joinpath("provenance.json").read_text())
        provenance["requested_ticker"] = ticker
        return calibrate(frame["close"], provenance)
    if not ticker or not ticker.strip():
        raise ValueError("a ticker is required for live calibration")
    symbol = ticker.strip().upper()
    if symbol[-1:].isdigit() and "." not in symbol:
        symbol += ".SA"
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ValueError("live data requires: pip install '.[market]' (or use --offline)") from exc
    try:
        frame = yf.download(
            symbol,
            period="5y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            multi_level_index=False,
        )
    except Exception as exc:
        raise ValueError("price download failed; retry or explicitly select --offline") from exc
    if frame is None or frame.empty or "Close" not in frame:
        raise ValueError("no daily prices returned; retry or explicitly select --offline")
    return calibrate(
        frame["Close"],
        {
            "kind": "live_adjusted_prices",
            "source": "Yahoo Finance via yfinance.download(auto_adjust=True)",
            "ticker": symbol,
            "retrieved_at_utc": datetime.now(UTC).isoformat(),
            "documentation": "https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html",
        },
    )
