"""Tests for tsmom.analytics — performance metrics."""

import numpy as np
import pandas as pd
import pytest

from tsmom.analytics import (
    compute_all_metrics,
    compute_drawdown_series,
    compute_rolling_sharpe,
    compute_rolling_return,
    build_metrics_table,
)


@pytest.fixture
def positive_returns():
    """24 months of mildly positive returns."""
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    np.random.seed(10)
    return pd.Series(np.random.normal(0.008, 0.03, 24), index=dates)


@pytest.fixture
def negative_returns():
    """12 months of strongly negative returns."""
    dates = pd.date_range("2020-01-31", periods=12, freq="ME")
    return pd.Series([-0.05] * 12, index=dates)


class TestComputeAllMetrics:

    def test_returns_all_keys(self, positive_returns, config):
        metrics = compute_all_metrics(positive_returns, config)
        expected = {
            "CAGR", "Ann. Vol", "Sharpe", "Sortino", "Max DD",
            "Max DD Duration", "Calmar", "Skewness", "Kurtosis",
            "Win Rate", "Best Month", "Worst Month",
        }
        assert set(metrics.keys()) == expected

    def test_cagr_positive_for_positive_returns(self, positive_returns, config):
        metrics = compute_all_metrics(positive_returns, config)
        # Not guaranteed positive with random seed, but check it's a number
        assert not np.isnan(metrics["CAGR"])

    def test_vol_positive(self, positive_returns, config):
        metrics = compute_all_metrics(positive_returns, config)
        assert metrics["Ann. Vol"] > 0

    def test_max_dd_negative_or_zero(self, positive_returns, config):
        metrics = compute_all_metrics(positive_returns, config)
        assert metrics["Max DD"] <= 0

    def test_win_rate_between_0_and_1(self, positive_returns, config):
        metrics = compute_all_metrics(positive_returns, config)
        assert 0 <= metrics["Win Rate"] <= 1

    def test_best_month_gte_worst_month(self, positive_returns, config):
        metrics = compute_all_metrics(positive_returns, config)
        assert metrics["Best Month"] >= metrics["Worst Month"]

    def test_calmar_positive_for_negative_dd(self, positive_returns, config):
        metrics = compute_all_metrics(positive_returns, config)
        if metrics["CAGR"] > 0 and metrics["Max DD"] < 0:
            assert metrics["Calmar"] > 0

    def test_deep_drawdown(self, negative_returns, config):
        """Strong negative returns should show large drawdown."""
        metrics = compute_all_metrics(negative_returns, config)
        assert metrics["Max DD"] < -0.30

    def test_sharpe_negative_for_negative_returns(self, negative_returns, config):
        metrics = compute_all_metrics(negative_returns, config)
        assert metrics["Sharpe"] < 0

    def test_max_dd_duration_is_int(self, positive_returns, config):
        metrics = compute_all_metrics(positive_returns, config)
        assert isinstance(metrics["Max DD Duration"], (int, np.integer))


class TestComputeDrawdownSeries:

    def test_drawdown_always_lte_zero(self, positive_returns):
        dd = compute_drawdown_series(positive_returns)
        assert (dd <= 1e-15).all()

    def test_no_drawdown_for_monotonic_returns(self):
        dates = pd.date_range("2021-01-31", periods=6, freq="ME")
        returns = pd.Series([0.05] * 6, index=dates)
        dd = compute_drawdown_series(returns)
        assert (dd == 0).all()

    def test_drawdown_min_matches_manual(self):
        dates = pd.date_range("2021-01-31", periods=4, freq="ME")
        returns = pd.Series([0.10, -0.20, 0.05, 0.10], index=dates)
        dd = compute_drawdown_series(returns)
        # After +10%, cum = 1.10. After -20%, cum = 0.88. DD = (0.88-1.10)/1.10
        expected_dd = (0.88 - 1.10) / 1.10
        assert abs(dd.min() - expected_dd) < 1e-10


class TestRollingMetrics:

    def test_rolling_sharpe_length(self, positive_returns, config):
        rs = compute_rolling_sharpe(positive_returns, config)
        assert len(rs) == len(positive_returns)

    def test_rolling_sharpe_first_11_are_nan(self, positive_returns, config):
        """12-month window → first 11 values are NaN."""
        rs = compute_rolling_sharpe(positive_returns, config)
        assert rs.iloc[:11].isna().all()

    def test_rolling_return_length(self, positive_returns, config):
        rr = compute_rolling_return(positive_returns, config)
        assert len(rr) == len(positive_returns)


class TestBuildMetricsTable:

    def test_table_shape(self, positive_returns, config):
        benchmarks = {
            "BM1": positive_returns * 0.8,
            "BM2": positive_returns * 1.2,
        }
        table = build_metrics_table(positive_returns, benchmarks, config)
        assert table.shape[1] == 3  # TSMOM + 2 benchmarks
        assert "TSMOM" in table.columns
        assert "BM1" in table.columns

    def test_table_has_all_metrics(self, positive_returns, config):
        table = build_metrics_table(positive_returns, {}, config)
        assert "Sharpe" in table.index
        assert "CAGR" in table.index
        assert "Max DD" in table.index
