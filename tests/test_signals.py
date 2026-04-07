"""Tests for tsmom.signals — momentum signal computation."""

import numpy as np
import pandas as pd
import pytest

from tsmom.signals import (
    compute_momentum_signal,
    compute_signal_at_dates,
    get_rebalance_dates,
)


class TestComputeMomentumSignal:
    """Test the 12-1 momentum signal computation."""

    def test_output_shape_matches_prices(self, daily_prices, config):
        signals = compute_momentum_signal(daily_prices, config)
        assert signals.shape == daily_prices.shape

    def test_signal_values_are_valid(self, daily_prices, config):
        """Signals must be +1, -1, or 0 only."""
        signals = compute_momentum_signal(daily_prices, config)
        unique_vals = set(signals.values.flatten())
        assert unique_vals.issubset({-1, 0, 1})

    def test_early_rows_are_zero(self, daily_prices, config):
        """First 252 rows lack sufficient history — signals must be 0."""
        signals = compute_momentum_signal(daily_prices, config)
        lookback = config["signal"]["lookback_days"]
        early = signals.iloc[:lookback]
        assert (early == 0).all().all()

    def test_positive_trend_gives_long_signal(self, config):
        """Steadily rising prices should produce +1 signal."""
        dates = pd.bdate_range("2018-01-02", periods=400, freq="B")
        prices = pd.DataFrame(
            {"A": np.linspace(100, 200, 400)}, index=dates
        )
        signals = compute_momentum_signal(prices, config)
        # After enough history, signal should be +1
        late_signals = signals["A"].iloc[300:]
        assert (late_signals == 1).all()

    def test_negative_trend_gives_short_signal(self, config):
        """Steadily falling prices should produce -1 signal."""
        dates = pd.bdate_range("2018-01-02", periods=400, freq="B")
        prices = pd.DataFrame(
            {"A": np.linspace(200, 50, 400)}, index=dates
        )
        signals = compute_momentum_signal(prices, config)
        late_signals = signals["A"].iloc[300:]
        assert (late_signals == -1).all()

    def test_custom_lookback_and_skip(self):
        """Verify signal respects custom lookback/skip config."""
        cfg = {"signal": {"lookback_days": 10, "skip_days": 2}}
        dates = pd.bdate_range("2023-01-02", periods=30, freq="B")
        # Rising first 20, then falling
        prices = pd.DataFrame(
            {"X": list(range(100, 130))}, index=dates, dtype=float
        )
        signals = compute_momentum_signal(prices, cfg)
        # After day 10, signal should be available and +1 (still rising)
        assert signals["X"].iloc[15] == 1


class TestComputeSignalAtDates:
    """Test signal extraction at specific rebalance dates."""

    def test_returns_only_requested_dates(self, daily_prices, config):
        rebal = get_rebalance_dates(daily_prices)
        signals = compute_signal_at_dates(daily_prices, rebal, config)
        assert len(signals) == len(rebal)
        assert signals.index.equals(rebal)

    def test_values_match_full_signal(self, daily_prices, config):
        """Spot-check: signal at rebalance date matches full daily signal."""
        full_signals = compute_momentum_signal(daily_prices, config)
        rebal = get_rebalance_dates(daily_prices)
        at_dates = compute_signal_at_dates(daily_prices, rebal, config)

        # For dates that exist in the daily index, values should match
        common = rebal.intersection(full_signals.index)
        pd.testing.assert_frame_equal(
            at_dates.loc[common], full_signals.loc[common]
        )


class TestGetRebalanceDates:
    """Test month-end rebalance date extraction."""

    def test_returns_datetimeindex(self, daily_prices):
        dates = get_rebalance_dates(daily_prices)
        assert isinstance(dates, pd.DatetimeIndex)

    def test_one_date_per_month(self, daily_prices):
        dates = get_rebalance_dates(daily_prices)
        periods = dates.to_period("M")
        assert len(periods) == len(periods.unique())

    def test_dates_are_last_trading_day(self, daily_prices):
        """Each rebalance date should be the last business day in its month."""
        dates = get_rebalance_dates(daily_prices)
        for d in dates:
            month_data = daily_prices.loc[daily_prices.index.to_period("M") == d.to_period("M")]
            assert d == month_data.index[-1]

    def test_count_roughly_matches_months(self, daily_prices):
        """600 bdays ≈ 2.4 years ≈ 28-29 months."""
        dates = get_rebalance_dates(daily_prices)
        assert 25 <= len(dates) <= 35
