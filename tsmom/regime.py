"""
Regime overlay for scaling positions in crisis environments.

Two methods:
1. VIX threshold (default): if VIX > threshold → scale positions by crisis_scale.
2. HMM-based: fit 2-state HMM on SPY returns, high-vol state → scale down.

Off by default, pure TSMOM first.

Simplifying assumption: VIX fetched from yfinance (^VIX), may have gaps, forward-fill used.
"""

import numpy as np
import pandas as pd
import yfinance as yf


def apply_regime_overlay(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    config: dict,
) -> pd.DataFrame:
    """Apply regime overlay to scale positions in crisis regimes.

    No-op if regime_overlay.enabled is false.

    Args:
        weights: Capped weight DataFrame at rebalance dates.
        prices: Daily price DataFrame (used for HMM method).
        rebalance_dates: Rebalance dates.
        config: Full config dict.

    Returns:
        Adjusted weight DataFrame (scaled in crisis periods).
    """
    overlay_cfg = config["regime_overlay"]

    if not overlay_cfg["enabled"]:
        return weights

    method = overlay_cfg["method"]

    if method == "vix":
        return _apply_vix_overlay(weights, rebalance_dates, overlay_cfg)
    elif method == "hmm":
        return _apply_hmm_overlay(weights, prices, rebalance_dates, overlay_cfg)
    else:
        raise ValueError(f"Unknown regime method: {method}. Use 'vix' or 'hmm'.")


def _apply_vix_overlay(
    weights: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    overlay_cfg: dict,
) -> pd.DataFrame:
    """Scale positions down when VIX exceeds threshold.

    Args:
        weights: Weight DataFrame.
        rebalance_dates: Rebalance dates.
        overlay_cfg: regime_overlay config section.

    Returns:
        Scaled weight DataFrame.
    """
    vix_ticker = overlay_cfg["vix_ticker"]
    threshold = overlay_cfg["vix_threshold"]
    crisis_scale = overlay_cfg["crisis_scale"]

    # Fetch VIX data
    vix_data = yf.download(vix_ticker, period="max", auto_adjust=True, progress=False)
    if isinstance(vix_data.columns, pd.MultiIndex):
        vix_close = vix_data["Close"].squeeze()
    else:
        vix_close = vix_data["Close"]

    # Forward-fill gaps in VIX
    vix_close = vix_close.ffill()

    # Align VIX to rebalance dates
    vix_at_rebal = vix_close.reindex(rebalance_dates, method="ffill")

    # Scale factor: crisis_scale if VIX > threshold, else 1.0
    scale = pd.Series(1.0, index=rebalance_dates)
    scale[vix_at_rebal > threshold] = crisis_scale

    # Apply scale to all weights
    adjusted = weights.multiply(scale, axis=0)

    return adjusted


def _apply_hmm_overlay(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    overlay_cfg: dict,
) -> pd.DataFrame:
    """Scale positions down in HMM-detected high-volatility regime.

    Uses an expanding-window HMM: at each rebalance date, fit the model only
    on SPY returns available up to that date. This avoids lookahead bias
    (fitting on full history would leak future regime information into past
    position sizing decisions).

    Args:
        weights: Weight DataFrame.
        prices: Daily price DataFrame (needs 'SPY' column).
        rebalance_dates: Rebalance dates.
        overlay_cfg: regime_overlay config section.

    Returns:
        Scaled weight DataFrame.
    """
    from hmmlearn.hmm import GaussianHMM

    crisis_scale = overlay_cfg["crisis_scale"]
    n_states = overlay_cfg["hmm_n_states"]
    min_obs = 504  # Require at least ~2 years of data before fitting

    spy_returns = prices["SPY"].pct_change().dropna()

    scale = pd.Series(1.0, index=rebalance_dates)

    for dt in rebalance_dates:
        # Expanding window: only use data up to and including this date
        hist = spy_returns.loc[:dt]
        if len(hist) < min_obs:
            continue  # Not enough data, scale stays 1.0

        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        returns_arr = hist.values.reshape(-1, 1)
        model.fit(returns_arr)

        states = model.predict(returns_arr)
        state_series = pd.Series(states, index=hist.index)

        # High-vol state = the one with higher variance
        state_vols = [hist[state_series == s].std() for s in range(n_states)]
        high_vol_state = np.argmax(state_vols)

        # Use the most recent predicted state for this rebalance date
        if states[-1] == high_vol_state:
            scale[dt] = crisis_scale

    adjusted = weights.multiply(scale, axis=0)

    return adjusted


def get_regime_labels(
    prices: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    config: dict,
) -> pd.Series:
    """Get regime labels at rebalance dates for reporting.

    Returns 'crisis' or 'normal' at each date.

    Args:
        prices: Daily prices.
        rebalance_dates: Rebalance dates.
        config: Full config dict.

    Returns:
        Series of regime labels. All 'normal' if overlay disabled.
    """
    overlay_cfg = config["regime_overlay"]

    if not overlay_cfg["enabled"]:
        return pd.Series("normal", index=rebalance_dates)

    if overlay_cfg["method"] == "vix":
        vix_data = yf.download(
            overlay_cfg["vix_ticker"], period="max", auto_adjust=True, progress=False
        )
        if isinstance(vix_data.columns, pd.MultiIndex):
            vix_close = vix_data["Close"].squeeze()
        else:
            vix_close = vix_data["Close"]
        vix_close = vix_close.ffill()
        vix_at_rebal = vix_close.reindex(rebalance_dates, method="ffill")
        labels = pd.Series("normal", index=rebalance_dates)
        labels[vix_at_rebal > overlay_cfg["vix_threshold"]] = "crisis"
        return labels

    # HMM, same logic as overlay but return labels
    return pd.Series("normal", index=rebalance_dates)
