"""Tests for tsmom.backtest — walk-forward backtest engine."""

import numpy as np
import pandas as pd
import pytest

from tsmom.backtest import run_backtest, get_strategy_date_range


class TestRunBacktest:
    """Test the full backtest pipeline with synthetic data."""

    def test_returns_expected_keys(self, daily_prices, config):
        results = run_backtest(daily_prices, config)
        expected_keys = {
            "weights", "weights_aligned", "portfolio_returns",
            "cumulative_returns", "costs", "monthly_asset_returns",
            "rebalance_dates", "signals", "daily_returns",
        }
        assert set(results.keys()) == expected_keys

    def test_portfolio_returns_is_series(self, daily_prices, config):
        results = run_backtest(daily_prices, config)
        assert isinstance(results["portfolio_returns"], pd.Series)

    def test_cumulative_returns_is_series(self, daily_prices, config):
        results = run_backtest(daily_prices, config)
        assert isinstance(results["cumulative_returns"], pd.Series)

    def test_cumulative_starts_positive(self, daily_prices, config):
        results = run_backtest(daily_prices, config)
        assert results["cumulative_returns"].iloc[0] > 0

    def test_cumulative_is_product_of_returns(self, daily_prices, config):
        results = run_backtest(daily_prices, config)
        expected = (1 + results["portfolio_returns"]).cumprod()
        pd.testing.assert_series_equal(
            results["cumulative_returns"], expected, check_names=False
        )

    def test_weights_columns_match_tickers(self, daily_prices, config):
        results = run_backtest(daily_prices, config)
        assert list(results["weights"].columns) == list(daily_prices.columns)

    def test_signals_values_valid(self, daily_prices, config):
        results = run_backtest(daily_prices, config)
        unique = set(results["signals"].values.flatten())
        assert unique.issubset({-1, 0, 1})

    def test_costs_non_negative(self, daily_prices, config):
        results = run_backtest(daily_prices, config)
        assert (results["costs"] >= 0).all()

    def test_no_costs_increases_returns(self, daily_prices, config):
        """Disabling costs should produce >= portfolio returns."""
        results_with = run_backtest(daily_prices, config)
        config["transaction_costs"]["enabled"] = False
        results_without = run_backtest(daily_prices, config)

        # Cumulative return without costs should be >= with costs
        assert results_without["cumulative_returns"].iloc[-1] >= \
               results_with["cumulative_returns"].iloc[-1] - 1e-10

    def test_portfolio_returns_length(self, daily_prices, config):
        """Should have multiple monthly returns."""
        results = run_backtest(daily_prices, config)
        assert len(results["portfolio_returns"]) >= 5


class TestGetStrategyDateRange:

    def test_returns_tuple(self, daily_prices, config):
        results = run_backtest(daily_prices, config)
        start, end = get_strategy_date_range(results)
        assert isinstance(start, pd.Timestamp)
        assert isinstance(end, pd.Timestamp)
        assert end > start
