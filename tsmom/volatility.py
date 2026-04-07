"""
Realized volatility estimation for position sizing.

Two methods:
1. EWMA (default): exponentially weighted, halflife configurable (default 60 days).
2. Simple rolling: fixed window (default 63 days).

Both annualized by sqrt(252).

Simplifying assumption: EWMA vol uses daily returns, not intraday.
"""

import numpy as np
import pandas as pd


def compute_realized_vol(daily_returns: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compute annualized realized volatility for each asset.

    Dispatches to EWMA or rolling based on config.

    Args:
        daily_returns: Daily simple returns (DatetimeIndex, one col per ticker).
        config: Full config dict (uses volatility section).

    Returns:
        DataFrame of annualized realized vol, same shape as daily_returns.
    """
    method = config["volatility"]["method"]
    ann_factor = config["volatility"]["annualization_factor"]

    if method == "ewma":
        return _compute_ewma_vol(daily_returns, config, ann_factor)
    elif method == "rolling":
        return _compute_rolling_vol(daily_returns, config, ann_factor)
    else:
        raise ValueError(f"Unknown volatility method: {method}. Use 'ewma' or 'rolling'.")


def _compute_ewma_vol(
    daily_returns: pd.DataFrame, config: dict, ann_factor: int
) -> pd.DataFrame:
    """EWMA volatility: ewm(returns, halflife).std() * sqrt(ann_factor).

    Args:
        daily_returns: Daily returns DataFrame.
        config: Config dict.
        ann_factor: Annualization factor (252).

    Returns:
        Annualized EWMA vol DataFrame.
    """
    halflife = config["volatility"]["ewma_halflife"]

    vol = daily_returns.ewm(halflife=halflife).std() * np.sqrt(ann_factor)

    return vol


def _compute_rolling_vol(
    daily_returns: pd.DataFrame, config: dict, ann_factor: int
) -> pd.DataFrame:
    """Rolling window volatility: rolling(window).std() * sqrt(ann_factor).

    Args:
        daily_returns: Daily returns DataFrame.
        config: Config dict.
        ann_factor: Annualization factor (252).

    Returns:
        Annualized rolling vol DataFrame.
    """
    window = config["volatility"]["rolling_window"]

    vol = daily_returns.rolling(window).std() * np.sqrt(ann_factor)

    return vol


def get_vol_at_dates(
    daily_returns: pd.DataFrame, rebalance_dates: pd.DatetimeIndex, config: dict
) -> pd.DataFrame:
    """Compute realized vol and extract values at rebalance dates.

    Args:
        daily_returns: Daily returns DataFrame.
        rebalance_dates: Dates to extract vol for.
        config: Full config dict.

    Returns:
        Vol DataFrame at rebalance dates only.
    """
    vol = compute_realized_vol(daily_returns, config)

    # Align to rebalance dates (use last available vol on or before each date)
    vol_at_dates = vol.reindex(rebalance_dates, method="ffill")

    return vol_at_dates
