"""
Walk-forward backtest engine.

Computes monthly portfolio returns by combining weights with asset returns,
deducting transaction costs, and building a cumulative equity curve.

No lookahead bias: all signals computed using only data available at rebalance date.
Universe is fixed (all 13 ETFs), no dynamic selection — but this uses current ETFs
selected ex-post, which embeds inception bias vs. the Moskowitz et al. futures setup.
This is a known limitation of an ETF-based implementation.

Simplifying assumptions:
- Rebalance at close prices on last trading day of month.
- No cash interest earned on un-invested capital (if gross < 1.0).
"""

import numpy as np
import pandas as pd

from tsmom.loader import compute_returns, get_monthly_returns
from tsmom.signals import compute_signal_at_dates, get_rebalance_dates
from tsmom.volatility import get_vol_at_dates
from tsmom.portfolio import build_weight_history
from tsmom.costs import compute_transaction_costs
from tsmom.regime import apply_regime_overlay


def run_backtest(prices: pd.DataFrame, config: dict) -> dict:
    """Run the full TSMOM backtest pipeline.

    Steps:
      1. Compute daily returns and monthly returns.
      2. Identify rebalance dates (month-end).
      3. Compute 12-1 momentum signals at rebalance dates.
      4. Compute realized vol at rebalance dates.
      5. Build vol-targeted, capped weights.
      6. Apply regime overlay (if enabled).
      7. Compute transaction costs.
      8. Compute monthly portfolio returns.
      9. Build cumulative equity curve.

    Args:
        prices: Daily adjusted close prices for all assets.
        config: Full config dict.

    Returns:
        Dict with keys:
          - 'weights': DataFrame of weights at rebalance dates
          - 'portfolio_returns': Series of monthly portfolio returns
          - 'cumulative_returns': Series of cumulative portfolio value (starts at 1.0)
          - 'costs': Series of transaction costs per rebalance
          - 'monthly_asset_returns': DataFrame of monthly asset returns
          - 'rebalance_dates': DatetimeIndex
          - 'signals': DataFrame of signals at rebalance dates
    """
    # Daily returns for vol estimation
    daily_returns = compute_returns(prices)

    # Monthly returns for P&L
    monthly_returns = get_monthly_returns(prices)

    # Rebalance dates = last trading day of each month
    rebalance_dates = get_rebalance_dates(prices)

    # Compute signals at rebalance dates
    signals = compute_signal_at_dates(prices, rebalance_dates, config)

    # Compute realized vol at rebalance dates
    realized_vol = get_vol_at_dates(daily_returns, rebalance_dates, config)

    # Build weights: raw → capped
    weights = build_weight_history(signals, realized_vol, config)

    # Apply regime overlay (no-op if disabled)
    weights = apply_regime_overlay(weights, prices, rebalance_dates, config)

    # Transaction costs
    costs = compute_transaction_costs(weights, config)

    # Align weights and monthly returns
    # Weights at month-end t determine returns earned in month t+1
    # So shift weights forward by 1 period to align with next month's returns
    common_dates = weights.index.intersection(monthly_returns.index)
    weights_aligned = weights.loc[common_dates]
    returns_aligned = monthly_returns.loc[common_dates]

    # Portfolio return at each rebalance: sum(w_i * r_i) - cost
    # Weight at t-1 is applied to returns at t
    # We shift weights back: weight from prior month applies to current month return
    w_shifted = weights_aligned.shift(1)
    costs_shifted = costs.reindex(common_dates).shift(1)

    # Drop first row (no prior weights)
    w_shifted = w_shifted.iloc[1:]
    r_period = returns_aligned.iloc[1:]
    costs_shifted = costs_shifted.iloc[1:].fillna(0)

    # Monthly portfolio return
    portfolio_returns = (w_shifted * r_period).sum(axis=1) - costs_shifted
    portfolio_returns.name = "TSMOM"

    # Cumulative returns (growth of $1)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    cumulative_returns.name = "TSMOM"

    return {
        "weights": weights,
        "weights_aligned": w_shifted,  # Shifted weights aligned to returns (for attribution)
        "portfolio_returns": portfolio_returns,
        "cumulative_returns": cumulative_returns,
        "costs": costs,
        "monthly_asset_returns": r_period,  # Returns aligned to shifted weights
        "rebalance_dates": rebalance_dates,
        "signals": signals,
        "daily_returns": daily_returns,
    }


def get_strategy_date_range(backtest_results: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Get the effective start and end dates of the backtest.

    Args:
        backtest_results: Output dict from run_backtest.

    Returns:
        (start_date, end_date) tuple.
    """
    cum_ret = backtest_results["cumulative_returns"]
    return cum_ret.index[0], cum_ret.index[-1]
