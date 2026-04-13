"""
Time-series momentum signal computation.

Implements the 12-1 momentum signal from Moskowitz, Ooi & Pedersen (2012):
  signal(t) = sign( cumulative_return(t-252, t-21) )

Lookback: 12 months (~252 trading days).
Skip: most recent 1 month (~21 trading days), avoids short-term mean reversion.
Output: +1 (long) or -1 (short). 0 if insufficient history.
"""

import numpy as np
import pandas as pd


def compute_momentum_signal(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compute 12-1 momentum signal for each asset on each day.

    Args:
        prices: Daily adjusted close prices (DatetimeIndex, one col per ticker).
        config: Full config dict (uses signal.lookback_days, signal.skip_days).

    Returns:
        DataFrame of signals: +1 (long), -1 (short), 0 (no data).
        Same shape as prices.
    """
    lookback = config["signal"]["lookback_days"]
    skip = config["signal"]["skip_days"]

    # Cumulative return from t-lookback to t-skip
    # cum_ret = price[t-skip] / price[t-lookback] - 1
    price_end = prices.shift(skip)        # Price at t-skip
    price_start = prices.shift(lookback)  # Price at t-lookback

    cum_return = price_end / price_start - 1

    # Signal is the sign of the cumulative return
    signal = np.sign(cum_return)

    # Replace NaN with 0 (insufficient history → no position)
    signal = signal.fillna(0).astype(int)

    return signal


def compute_signal_at_dates(
    prices: pd.DataFrame, rebalance_dates: pd.DatetimeIndex, config: dict
) -> pd.DataFrame:
    """Compute momentum signals only at specific rebalance dates.

    More efficient than computing daily signals when only month-end is needed.

    Args:
        prices: Daily adjusted close prices.
        rebalance_dates: DatetimeIndex of rebalance dates.
        config: Full config dict.

    Returns:
        DataFrame of signals at rebalance dates only.
    """
    all_signals = compute_momentum_signal(prices, config)

    # Align to rebalance dates (use last available signal on or before each date)
    signals_at_dates = all_signals.reindex(rebalance_dates, method="ffill")

    return signals_at_dates


def get_rebalance_dates(prices: pd.DataFrame) -> pd.DatetimeIndex:
    """Extract month-end rebalance dates from the price index.

    Uses the last trading day of each month (the actual last date in the data
    for that month, not the calendar month-end).

    Args:
        prices: Daily price DataFrame.

    Returns:
        DatetimeIndex of month-end trading dates.
    """
    # Group by year-month, take last index value in each group
    monthly_groups = prices.groupby(prices.index.to_period("M"))
    rebalance_dates = pd.DatetimeIndex([group.index[-1] for _, group in monthly_groups])

    return rebalance_dates
