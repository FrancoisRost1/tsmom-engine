"""Tests for tsmom.volatility — realized volatility estimation."""

import numpy as np
import pandas as pd
import pytest

from tsmom.volatility import compute_realized_vol, get_vol_at_dates
from tsmom.signals import get_rebalance_dates


class TestComputeRealizedVol:
    """Test EWMA and rolling vol estimation."""

    def test_ewma_output_shape(self, daily_returns, config):
        vol = compute_realized_vol(daily_returns, config)
        assert vol.shape == daily_returns.shape

    def test_rolling_output_shape(self, daily_returns, config):
        config["volatility"]["method"] = "rolling"
        vol = compute_realized_vol(daily_returns, config)
        assert vol.shape == daily_returns.shape

    def test_vol_is_positive(self, daily_returns, config):
        """Realized vol should be non-negative everywhere (NaN for early rows)."""
        vol = compute_realized_vol(daily_returns, config)
        valid = vol.dropna()
        assert (valid >= 0).all().all()

    def test_vol_is_annualized(self, daily_returns, config):
        """Vol should be in annualized range (roughly 5%-50% for typical assets)."""
        vol = compute_realized_vol(daily_returns, config)
        median_vol = vol.iloc[-100:].median().median()
        # Synthetic data has ~1.2% daily std → ~19% annualized
        assert 0.05 < median_vol < 0.60

    def test_higher_vol_asset_gets_higher_estimate(self, config):
        """Asset with larger return dispersion should have higher vol."""
        np.random.seed(7)
        dates = pd.bdate_range("2020-01-02", periods=200, freq="B")
        low_vol = pd.Series(np.random.normal(0, 0.005, 200), index=dates, name="LO")
        high_vol = pd.Series(np.random.normal(0, 0.025, 200), index=dates, name="HI")
        returns = pd.concat([low_vol, high_vol], axis=1)

        vol = compute_realized_vol(returns, config)
        last = vol.iloc[-1]
        assert last["HI"] > last["LO"]

    def test_unknown_method_raises(self, daily_returns, config):
        config["volatility"]["method"] = "garman_klass"
        with pytest.raises(ValueError, match="Unknown volatility method"):
            compute_realized_vol(daily_returns, config)

    def test_ewma_vs_rolling_differ(self, daily_returns, config):
        """EWMA and rolling should produce different values."""
        vol_ewma = compute_realized_vol(daily_returns, config)
        config["volatility"]["method"] = "rolling"
        vol_rolling = compute_realized_vol(daily_returns, config)
        # Not identical (different algorithms)
        assert not np.allclose(
            vol_ewma.iloc[-1].values,
            vol_rolling.iloc[-1].values,
            atol=1e-6,
        )


class TestGetVolAtDates:
    """Test vol extraction at rebalance dates."""

    def test_output_index_matches_rebalance_dates(self, daily_prices, daily_returns, config):
        rebal = get_rebalance_dates(daily_prices)
        vol = get_vol_at_dates(daily_returns, rebal, config)
        assert vol.index.equals(rebal)

    def test_output_columns_match_returns(self, daily_prices, daily_returns, config):
        rebal = get_rebalance_dates(daily_prices)
        vol = get_vol_at_dates(daily_returns, rebal, config)
        assert list(vol.columns) == list(daily_returns.columns)
