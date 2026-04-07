"""
Portfolio construction: vol-targeted weights, position caps, and rebalancing.

Position sizing formula:
  raw_weight(i,t) = signal(i,t) * (target_vol / realized_vol(i,t))

Cap application order:
  1. Compute raw weights from vol targeting.
  2. Clip per-asset to ±max_asset_leverage.
  3. If gross leverage > max_portfolio_leverage → pro-rata scale-down.

Simplifying assumptions:
- Shorting via negative weight — assumes ETFs are shortable at no extra cost.
- No margin requirements modeled.
- No cash interest earned on un-invested capital.
"""

import numpy as np
import pandas as pd


def compute_raw_weights(
    signals: pd.DataFrame, realized_vol: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """Compute raw vol-targeted weights before capping.

    raw_weight = signal * (target_vol / realized_vol)

    Where realized_vol is NaN or zero, weight is 0 (no position).

    Args:
        signals: Signal DataFrame at rebalance dates (+1, -1, or 0).
        realized_vol: Realized vol DataFrame at same rebalance dates.
        config: Full config dict.

    Returns:
        Raw weight DataFrame (same index/columns as signals).
    """
    target_vol = config["position_sizing"]["target_vol"]

    # Avoid division by zero or NaN vol
    safe_vol = realized_vol.replace(0, np.nan)

    raw_weights = signals * (target_vol / safe_vol)

    # Where vol was NaN → weight is 0
    raw_weights = raw_weights.fillna(0)

    return raw_weights


def apply_position_caps(weights: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply per-asset and portfolio-level leverage caps.

    Steps:
      1. Clip each weight to [-max_asset_leverage, +max_asset_leverage].
      2. If sum(|weights|) > max_portfolio_leverage, scale all down pro-rata.

    Args:
        weights: Raw weight DataFrame.
        config: Full config dict.

    Returns:
        Capped weight DataFrame.
    """
    max_asset = config["position_sizing"]["max_asset_leverage"]
    max_portfolio = config["position_sizing"]["max_portfolio_leverage"]

    # Step 1: Per-asset cap
    capped = weights.clip(lower=-max_asset, upper=max_asset)

    # Step 2: Portfolio-level cap
    gross_leverage = capped.abs().sum(axis=1)
    scale_factor = (max_portfolio / gross_leverage).clip(upper=1.0)

    # Multiply each row by its scale factor
    capped = capped.multiply(scale_factor, axis=0)

    return capped


def build_weight_history(
    signals: pd.DataFrame, realized_vol: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """Full pipeline: raw weights → caps → final weights at rebalance dates.

    Args:
        signals: Signal DataFrame at rebalance dates.
        realized_vol: Vol DataFrame at rebalance dates.
        config: Full config dict.

    Returns:
        Final capped weight DataFrame at rebalance dates.
    """
    raw = compute_raw_weights(signals, realized_vol, config)
    capped = apply_position_caps(raw, config)
    return capped


def compute_gross_leverage(weights: pd.DataFrame) -> pd.Series:
    """Compute gross leverage (sum of absolute weights) per date.

    Args:
        weights: Weight DataFrame.

    Returns:
        Series of gross leverage values.
    """
    return weights.abs().sum(axis=1)


def compute_net_exposure(weights: pd.DataFrame) -> pd.Series:
    """Compute net exposure (sum of signed weights) per date.

    Args:
        weights: Weight DataFrame.

    Returns:
        Series of net exposure values.
    """
    return weights.sum(axis=1)
