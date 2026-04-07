"""
Transaction cost model.

Flat basis point cost applied to turnover at each rebalance.
  cost(t) = sum(|w_new - w_old|) * cost_bps / 10000

Simplifying assumption: flat bps — no market impact, no bid-ask spread modeling.
"""

import numpy as np
import pandas as pd


def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    """Compute turnover at each rebalance date.

    Turnover = sum of absolute weight changes from previous rebalance.
    First rebalance assumes all positions start from zero.

    Args:
        weights: Weight DataFrame at rebalance dates.

    Returns:
        Series of turnover values per rebalance date.
    """
    # Shift gives prior period weights; first row starts from 0
    prev_weights = weights.shift(1).fillna(0)
    turnover = (weights - prev_weights).abs().sum(axis=1)
    return turnover


def compute_transaction_costs(weights: pd.DataFrame, config: dict) -> pd.Series:
    """Compute transaction cost deducted at each rebalance.

    Args:
        weights: Weight DataFrame at rebalance dates.
        config: Full config dict (uses transaction_costs section).

    Returns:
        Series of cost values per date. Zero if costs are disabled.
    """
    if not config["transaction_costs"]["enabled"]:
        return pd.Series(0.0, index=weights.index)

    cost_bps = config["transaction_costs"]["cost_bps"]
    turnover = compute_turnover(weights)

    costs = turnover * cost_bps / 10000

    return costs
