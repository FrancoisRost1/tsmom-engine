"""
Data loader: fetch ETF prices from yfinance with caching and validation.

Simplifying assumptions:
- ETF prices used as proxies for asset class exposure (no futures, no leverage cost).
- Ragged start: each asset enters when sufficient history is available.
"""

import pandas as pd
import yfinance as yf
from pathlib import Path


def fetch_prices(tickers: list[str], config: dict) -> pd.DataFrame:
    """Fetch adjusted close prices for all tickers via yfinance.

    Uses local CSV cache to avoid redundant downloads.
    Falls back to yfinance if cache is stale or missing.

    Args:
        tickers: List of ETF ticker symbols.
        config: Full config dict (uses paths.cache).

    Returns:
        DataFrame with DatetimeIndex and one column per ticker (adjusted close).
    """
    cache_dir = Path(config["paths"]["cache"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "prices.csv"

    # Try cache first
    if cache_path.exists():
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        # Use cache if it has all tickers and is recent (within 3 days)
        if set(tickers).issubset(cached.columns):
            last_date = cached.index.max()
            if (pd.Timestamp.now() - last_date).days <= 3:
                return cached[tickers]

    # Download from yfinance
    prices = _download_prices(tickers)

    # Save to cache
    prices.to_csv(cache_path)

    return prices


def _download_prices(tickers: list[str]) -> pd.DataFrame:
    """Download adjusted close prices from yfinance.

    Args:
        tickers: List of ticker symbols.

    Returns:
        DataFrame with DatetimeIndex, one column per ticker.
    """
    data = yf.download(tickers, period="max", auto_adjust=True, progress=False)

    # yf.download returns multi-level columns when multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        # Single ticker case
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})

    return prices


def validate_prices(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Validate and clean price data.

    - Drops rows where all values are NaN.
    - Forward-fills gaps (up to 5 days, e.g., holidays).
    - Logs which tickers have data and their date ranges.

    Args:
        prices: Raw price DataFrame from fetch_prices.
        tickers: Expected ticker list.

    Returns:
        Cleaned price DataFrame.
    """
    # Ensure all requested tickers are present
    missing = set(tickers) - set(prices.columns)
    if missing:
        raise ValueError(f"Missing tickers in price data: {missing}")

    # Drop fully empty rows
    prices = prices.dropna(how="all")

    # Sort by date
    prices = prices.sort_index()

    # Forward-fill small gaps (holidays, data gaps up to 5 days)
    prices = prices.ffill(limit=5)

    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily simple returns from prices.

    Args:
        prices: Adjusted close price DataFrame.

    Returns:
        DataFrame of daily simple returns (first row is NaN, dropped).
    """
    returns = prices.pct_change()
    returns = returns.iloc[1:]  # Drop first NaN row
    return returns


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from prices.

    Used for volatility estimation (log returns are additive).

    Args:
        prices: Adjusted close price DataFrame.

    Returns:
        DataFrame of daily log returns.
    """
    import numpy as np

    log_ret = np.log(prices / prices.shift(1))
    log_ret = log_ret.iloc[1:]
    return log_ret


def get_monthly_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Resample to month-end prices using actual last trading day of each month.

    Uses groupby on calendar month to pick the real last trading date,
    so the index contains actual trading dates (e.g. Jan 29) rather than
    calendar month-ends (Jan 31).  This keeps the index aligned with
    get_rebalance_dates() and avoids silent row drops on intersection.

    Args:
        prices: Daily price DataFrame.

    Returns:
        Month-end price DataFrame indexed by actual last trading day.
    """
    # Group by year-month, take last row per group, preserves DatetimeIndex
    return prices.groupby(prices.index.to_period("M")).tail(1)


def get_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly simple returns from daily prices.

    Args:
        prices: Daily adjusted close prices.

    Returns:
        Monthly simple returns DataFrame indexed by actual last trading day.
    """
    monthly_prices = get_monthly_prices(prices)
    monthly_returns = monthly_prices.pct_change().iloc[1:]
    return monthly_returns
