"""
Return attribution: asset class decomposition and long/short P&L split.

Asset class decomposition: contribution = sum(weight * return) per class.
Long/short split: separate P&L from long (+1) vs short (-1) positions.
Literature says shorts are key source of TSMOM alpha.
"""

import numpy as np
import pandas as pd

from utils.config_loader import get_asset_class_map


def compute_asset_class_attribution(
    weights: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Compute return contribution by asset class per month.

    Contribution(class, t) = sum over assets in class of (weight_i * return_i).

    Args:
        weights: Weight DataFrame at rebalance dates (shifted to align with returns).
        monthly_returns: Monthly asset return DataFrame.
        config: Full config dict.

    Returns:
        DataFrame with columns = asset classes, rows = months.
    """
    ac_map = get_asset_class_map(config)
    asset_classes = ["equities", "bonds", "commodities", "fx"]

    # Align weights and returns
    common = weights.index.intersection(monthly_returns.index)
    w = weights.loc[common]
    r = monthly_returns.loc[common]

    attribution = pd.DataFrame(index=common)

    for ac in asset_classes:
        tickers_in_class = [t for t, cls in ac_map.items() if cls == ac]
        available = [t for t in tickers_in_class if t in w.columns and t in r.columns]
        if available:
            attribution[ac] = (w[available] * r[available]).sum(axis=1)
        else:
            attribution[ac] = 0.0

    return attribution


def compute_cumulative_asset_class_attribution(
    weights: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Cumulative return contribution by asset class.

    Args:
        weights: Shifted weight DataFrame.
        monthly_returns: Monthly returns.
        config: Full config dict.

    Returns:
        DataFrame of cumulative contributions per asset class.
    """
    monthly_attr = compute_asset_class_attribution(weights, monthly_returns, config)
    return monthly_attr.cumsum()


def compute_long_short_attribution(
    weights: pd.DataFrame,
    monthly_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Split portfolio P&L into long-side and short-side contributions.

    Long contribution = sum(w_i * r_i) for all w_i > 0.
    Short contribution = sum(w_i * r_i) for all w_i < 0.

    Args:
        weights: Weight DataFrame (shifted to align with returns).
        monthly_returns: Monthly return DataFrame.

    Returns:
        DataFrame with columns 'Long' and 'Short'.
    """
    common = weights.index.intersection(monthly_returns.index)
    w = weights.loc[common]
    r = monthly_returns.loc[common]

    contrib = w * r

    long_mask = w > 0
    short_mask = w < 0

    long_contrib = contrib.where(long_mask, 0).sum(axis=1)
    short_contrib = contrib.where(short_mask, 0).sum(axis=1)

    result = pd.DataFrame({"Long": long_contrib, "Short": short_contrib}, index=common)
    return result


def compute_long_short_statistics(
    long_short_df: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """Compute Sharpe, return, vol for long and short sides separately.

    Args:
        long_short_df: DataFrame with 'Long' and 'Short' columns.
        config: Full config dict.

    Returns:
        DataFrame with metrics as rows, Long/Short as columns.
    """
    rf = config["analytics"]["risk_free_rate"]
    rf_monthly = (1 + rf) ** (1 / 12) - 1

    stats = {}
    for side in ["Long", "Short"]:
        returns = long_short_df[side]
        excess = returns - rf_monthly
        ann_ret = returns.mean() * 12
        ann_vol = returns.std() * np.sqrt(12)
        sharpe = (excess.mean() / returns.std() * np.sqrt(12)) if returns.std() > 0 else np.nan

        stats[side] = {
            "Ann. Return": ann_ret,
            "Ann. Vol": ann_vol,
            "Sharpe": sharpe,
            "Avg Monthly": returns.mean(),
            "Hit Rate": (returns > 0).mean(),
        }

    return pd.DataFrame(stats)


def compute_per_asset_attribution(
    weights: pd.DataFrame,
    monthly_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Compute cumulative return contribution per individual asset.

    Args:
        weights: Shifted weight DataFrame.
        monthly_returns: Monthly return DataFrame.

    Returns:
        DataFrame of cumulative contribution per asset.
    """
    common = weights.index.intersection(monthly_returns.index)
    w = weights.loc[common]
    r = monthly_returns.loc[common]

    per_asset_contrib = (w * r).cumsum()
    return per_asset_contrib
