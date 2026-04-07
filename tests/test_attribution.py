"""Tests for tsmom.attribution — asset class and long/short decomposition."""

import numpy as np
import pandas as pd
import pytest

from tsmom.attribution import (
    compute_asset_class_attribution,
    compute_cumulative_asset_class_attribution,
    compute_long_short_attribution,
    compute_long_short_statistics,
    compute_per_asset_attribution,
)


class TestAssetClassAttribution:

    def test_output_columns(self, sample_weights, sample_monthly_returns, config):
        attr = compute_asset_class_attribution(
            sample_weights, sample_monthly_returns, config
        )
        assert set(attr.columns) == {"equities", "bonds", "commodities", "fx"}

    def test_total_matches_portfolio_return(self, sample_weights, sample_monthly_returns, config):
        """Sum across asset classes should equal total portfolio contribution."""
        attr = compute_asset_class_attribution(
            sample_weights, sample_monthly_returns, config
        )
        total_attr = attr.sum(axis=1)
        # Manual: sum(w * r) across all assets
        common = sample_weights.index.intersection(sample_monthly_returns.index)
        expected = (sample_weights.loc[common] * sample_monthly_returns.loc[common]).sum(axis=1)
        pd.testing.assert_series_equal(total_attr, expected, check_names=False)

    def test_cumulative_is_cumsum(self, sample_weights, sample_monthly_returns, config):
        monthly = compute_asset_class_attribution(
            sample_weights, sample_monthly_returns, config
        )
        cumulative = compute_cumulative_asset_class_attribution(
            sample_weights, sample_monthly_returns, config
        )
        pd.testing.assert_frame_equal(cumulative, monthly.cumsum())


class TestLongShortAttribution:

    def test_output_columns(self, sample_weights, sample_monthly_returns):
        ls = compute_long_short_attribution(sample_weights, sample_monthly_returns)
        assert list(ls.columns) == ["Long", "Short"]

    def test_long_plus_short_equals_total(self, sample_weights, sample_monthly_returns):
        """Long + Short contributions should sum to total portfolio return."""
        ls = compute_long_short_attribution(sample_weights, sample_monthly_returns)
        total_ls = ls["Long"] + ls["Short"]

        common = sample_weights.index.intersection(sample_monthly_returns.index)
        expected = (sample_weights.loc[common] * sample_monthly_returns.loc[common]).sum(axis=1)
        pd.testing.assert_series_equal(total_ls, expected, check_names=False)

    def test_all_long_weights_produce_no_short(self):
        dates = pd.date_range("2021-01-31", periods=3, freq="ME")
        w = pd.DataFrame({"A": [0.5, 0.3, 0.2], "B": [0.3, 0.4, 0.5]}, index=dates)
        r = pd.DataFrame({"A": [0.01, 0.02, -0.01], "B": [0.02, -0.01, 0.03]}, index=dates)
        ls = compute_long_short_attribution(w, r)
        assert (ls["Short"] == 0).all()

    def test_all_short_weights_produce_no_long(self):
        dates = pd.date_range("2021-01-31", periods=3, freq="ME")
        w = pd.DataFrame({"A": [-0.5, -0.3, -0.2], "B": [-0.3, -0.4, -0.5]}, index=dates)
        r = pd.DataFrame({"A": [0.01, 0.02, -0.01], "B": [0.02, -0.01, 0.03]}, index=dates)
        ls = compute_long_short_attribution(w, r)
        assert (ls["Long"] == 0).all()


class TestLongShortStatistics:

    def test_output_columns(self, sample_weights, sample_monthly_returns, config):
        ls = compute_long_short_attribution(sample_weights, sample_monthly_returns)
        stats = compute_long_short_statistics(ls, config)
        assert set(stats.columns) == {"Long", "Short"}

    def test_output_rows(self, sample_weights, sample_monthly_returns, config):
        ls = compute_long_short_attribution(sample_weights, sample_monthly_returns)
        stats = compute_long_short_statistics(ls, config)
        expected_rows = {"Ann. Return", "Ann. Vol", "Sharpe", "Avg Monthly", "Hit Rate"}
        assert set(stats.index) == expected_rows


class TestPerAssetAttribution:

    def test_output_shape(self, sample_weights, sample_monthly_returns):
        pa = compute_per_asset_attribution(sample_weights, sample_monthly_returns)
        common = sample_weights.index.intersection(sample_monthly_returns.index)
        assert pa.shape == (len(common), len(sample_weights.columns))

    def test_is_cumulative(self, sample_weights, sample_monthly_returns):
        pa = compute_per_asset_attribution(sample_weights, sample_monthly_returns)
        common = sample_weights.index.intersection(sample_monthly_returns.index)
        expected = (sample_weights.loc[common] * sample_monthly_returns.loc[common]).cumsum()
        pd.testing.assert_frame_equal(pa, expected)
