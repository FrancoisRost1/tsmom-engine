"""
Shared fixtures for all TSMOM tests.

All data is synthetic — no yfinance calls.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Minimal config dict matching config.yaml structure
# ---------------------------------------------------------------------------
@pytest.fixture
def config():
    return {
        "universe": {
            "equities": ["SPY", "EFA"],
            "bonds": ["TLT"],
            "commodities": ["GLD"],
            "fx": ["UUP"],
        },
        "signal": {
            "lookback_days": 252,
            "skip_days": 21,
        },
        "volatility": {
            "method": "ewma",
            "ewma_halflife": 60,
            "rolling_window": 63,
            "annualization_factor": 252,
        },
        "position_sizing": {
            "target_vol": 0.10,
            "max_asset_leverage": 2.0,
            "max_portfolio_leverage": 3.0,
        },
        "rebalancing": {"frequency": "monthly"},
        "transaction_costs": {
            "enabled": True,
            "cost_bps": 10,
        },
        "regime_overlay": {
            "enabled": False,
            "method": "vix",
            "vix_ticker": "^VIX",
            "vix_threshold": 25,
            "crisis_scale": 0.5,
            "hmm_n_states": 2,
        },
        "benchmarks": {
            "spy": {"ticker": "SPY"},
            "sixty_forty": {
                "equity_ticker": "SPY",
                "bond_ticker": "AGG",
                "equity_weight": 0.60,
                "bond_weight": 0.40,
            },
            "equal_weight": {"use_universe": True},
        },
        "backtest": {
            "start_date": "max",
            "end_date": "latest",
            "initial_capital": 1000000,
        },
        "analytics": {
            "risk_free_rate": 0.04,
            "rolling_window_months": 12,
        },
        "memo": {
            "strong_sharpe": 1.0,
            "moderate_sharpe": 0.5,
            "max_dd_warning": -0.25,
        },
        "paths": {
            "raw_data": "data/raw",
            "processed_data": "data/processed",
            "cache": "data/cache",
            "outputs": "outputs",
        },
    }


# ---------------------------------------------------------------------------
# Synthetic daily prices — 600 trading days, 5 assets with upward drift
# ---------------------------------------------------------------------------
@pytest.fixture
def daily_prices():
    """~2.4 years of synthetic daily prices for 5 assets."""
    np.random.seed(42)
    n_days = 600
    dates = pd.bdate_range("2020-01-02", periods=n_days, freq="B")
    tickers = ["SPY", "EFA", "TLT", "GLD", "UUP"]

    # Geometric Brownian Motion with small drift
    daily_log_returns = np.random.normal(0.0003, 0.012, (n_days, len(tickers)))
    prices = 100 * np.exp(np.cumsum(daily_log_returns, axis=0))

    return pd.DataFrame(prices, index=dates, columns=tickers)


# ---------------------------------------------------------------------------
# Daily simple returns derived from daily_prices
# ---------------------------------------------------------------------------
@pytest.fixture
def daily_returns(daily_prices):
    ret = daily_prices.pct_change().iloc[1:]
    return ret


# ---------------------------------------------------------------------------
# Monthly returns derived from daily_prices
# ---------------------------------------------------------------------------
@pytest.fixture
def monthly_returns(daily_prices):
    monthly_prices = daily_prices.resample("ME").last()
    return monthly_prices.pct_change().iloc[1:]


# ---------------------------------------------------------------------------
# Small weight DataFrame (5 rebalance dates, 5 assets)
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_weights():
    dates = pd.date_range("2021-01-31", periods=5, freq="ME")
    data = {
        "SPY": [0.5, 0.6, -0.3, 0.4, 0.7],
        "EFA": [-0.2, 0.3, 0.4, -0.5, 0.1],
        "TLT": [0.3, -0.1, 0.2, 0.3, -0.2],
        "GLD": [0.1, 0.2, -0.1, 0.2, 0.3],
        "UUP": [-0.1, 0.1, 0.3, -0.1, 0.2],
    }
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# Matching monthly returns for sample_weights
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_monthly_returns():
    dates = pd.date_range("2021-01-31", periods=5, freq="ME")
    np.random.seed(99)
    data = np.random.normal(0.01, 0.03, (5, 5))
    return pd.DataFrame(
        data, index=dates, columns=["SPY", "EFA", "TLT", "GLD", "UUP"]
    )
