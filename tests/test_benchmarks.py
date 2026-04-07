"""Tests for tsmom.benchmarks — benchmark construction."""

import numpy as np
import pandas as pd
import pytest

from tsmom.benchmarks import (
    build_all_benchmarks,
    build_benchmark_cumulative,
    _build_spy_benchmark,
    _build_equal_weight_benchmark,
    _build_sixty_forty_benchmark,
)


@pytest.fixture
def strategy_dates(daily_prices):
    """Monthly dates aligned to the synthetic price data (actual last trading day)."""
    from tsmom.loader import get_monthly_returns
    monthly = get_monthly_returns(daily_prices)
    return monthly.index


class TestBuildSpyBenchmark:

    def test_returns_series(self, daily_prices, strategy_dates, config):
        result = _build_spy_benchmark(daily_prices, strategy_dates, config)
        assert isinstance(result, pd.Series)

    def test_length_matches_strategy(self, daily_prices, strategy_dates, config):
        result = _build_spy_benchmark(daily_prices, strategy_dates, config)
        assert len(result) <= len(strategy_dates)
        assert len(result) > 0


class TestBuildEqualWeightBenchmark:

    def test_returns_series(self, daily_prices, strategy_dates, config):
        result = _build_equal_weight_benchmark(daily_prices, strategy_dates, config)
        assert isinstance(result, pd.Series)

    def test_is_mean_of_complete_asset_returns(self, daily_prices, strategy_dates, config):
        """Equal-weight = mean across all assets for months where all have data."""
        result = _build_equal_weight_benchmark(daily_prices, strategy_dates, config)
        assert isinstance(result, pd.Series)
        assert len(result) > 0
        # With synthetic data (no NaNs), should be close to simple mean
        from tsmom.loader import get_monthly_returns
        monthly_ret = get_monthly_returns(daily_prices)
        complete = monthly_ret.dropna()
        expected = complete.mean(axis=1)
        # Align to common dates
        common = result.index.intersection(expected.index)
        pd.testing.assert_series_equal(result.loc[common], expected.loc[common], check_names=False)


class TestBuildSixtyFortyBenchmark:

    def test_returns_series(self, daily_prices, strategy_dates, config, monkeypatch):
        """60/40 benchmark should return a Series (mock AGG download)."""
        # Mock yf.download for AGG
        agg_dates = daily_prices.index
        mock_agg = pd.DataFrame(
            {"Close": np.linspace(100, 110, len(agg_dates))}, index=agg_dates
        )

        import tsmom.benchmarks as bm_mod
        monkeypatch.setattr(bm_mod.yf, "download", lambda *a, **kw: mock_agg)

        result = _build_sixty_forty_benchmark(daily_prices, strategy_dates, config)
        assert isinstance(result, pd.Series)
        assert len(result) > 0


class TestBuildAllBenchmarks:

    def test_returns_three_benchmarks(self, daily_prices, strategy_dates, config, monkeypatch):
        agg_dates = daily_prices.index
        mock_agg = pd.DataFrame(
            {"Close": np.linspace(100, 110, len(agg_dates))}, index=agg_dates
        )

        import tsmom.benchmarks as bm_mod
        monkeypatch.setattr(bm_mod.yf, "download", lambda *a, **kw: mock_agg)

        benchmarks = build_all_benchmarks(daily_prices, strategy_dates, config)
        assert set(benchmarks.keys()) == {"SPY", "60/40", "Equal Weight"}

    def test_all_series(self, daily_prices, strategy_dates, config, monkeypatch):
        agg_dates = daily_prices.index
        mock_agg = pd.DataFrame(
            {"Close": np.linspace(100, 110, len(agg_dates))}, index=agg_dates
        )

        import tsmom.benchmarks as bm_mod
        monkeypatch.setattr(bm_mod.yf, "download", lambda *a, **kw: mock_agg)

        benchmarks = build_all_benchmarks(daily_prices, strategy_dates, config)
        for name, series in benchmarks.items():
            assert isinstance(series, pd.Series), f"{name} is not a Series"


class TestBuildBenchmarkCumulative:

    def test_cumulative_starts_near_one(self):
        returns = {
            "A": pd.Series([0.01, 0.02, -0.01], index=pd.date_range("2021-01-31", periods=3, freq="ME")),
        }
        cum = build_benchmark_cumulative(returns)
        assert abs(cum["A"].iloc[0] - 1.01) < 1e-10

    def test_cumulative_monotonic_for_positive_returns(self):
        returns = {
            "A": pd.Series([0.01, 0.02, 0.03], index=pd.date_range("2021-01-31", periods=3, freq="ME")),
        }
        cum = build_benchmark_cumulative(returns)
        assert cum["A"].is_monotonic_increasing
