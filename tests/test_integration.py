"""Integration tests — end-to-end pipeline with synthetic data."""

import numpy as np
import pandas as pd
import pytest

from tsmom.backtest import run_backtest, get_strategy_date_range
from tsmom.benchmarks import build_all_benchmarks, build_benchmark_cumulative
from tsmom.analytics import compute_all_metrics, build_metrics_table
from tsmom.attribution import (
    compute_asset_class_attribution,
    compute_long_short_attribution,
)
from tsmom.reporter import generate_memo, format_metrics_table


class TestFullPipeline:
    """End-to-end: backtest → benchmarks → analytics → attribution → memo."""

    def test_pipeline_runs_without_error(self, daily_prices, config, monkeypatch):
        # Mock AGG download for benchmarks
        mock_agg = pd.DataFrame(
            {"Close": np.linspace(100, 110, len(daily_prices))},
            index=daily_prices.index,
        )
        import tsmom.benchmarks as bm_mod
        monkeypatch.setattr(bm_mod.yf, "download", lambda *a, **kw: mock_agg)

        # 1. Run backtest
        results = run_backtest(daily_prices, config)

        # 2. Build benchmarks
        strategy_dates = results["portfolio_returns"].index
        benchmark_returns = build_all_benchmarks(daily_prices, strategy_dates, config)

        # 3. Compute metrics
        strategy_metrics = compute_all_metrics(results["portfolio_returns"], config)
        metrics_table = build_metrics_table(
            results["portfolio_returns"], benchmark_returns, config
        )

        # 4. Attribution
        w_shifted = results["weights"].shift(1).iloc[1:]
        monthly_ret = results["monthly_asset_returns"]
        ac_attr = compute_asset_class_attribution(w_shifted, monthly_ret, config)
        ls_attr = compute_long_short_attribution(w_shifted, monthly_ret)

        # 5. Memo
        bm_metrics = {
            name: compute_all_metrics(ret, config)
            for name, ret in benchmark_returns.items()
        }
        memo = generate_memo(strategy_metrics, bm_metrics, config)

        # Assertions
        assert len(results["portfolio_returns"]) > 0
        assert set(benchmark_returns.keys()) == {"SPY", "60/40", "Equal Weight"}
        assert "Sharpe" in strategy_metrics
        assert ac_attr.shape[1] == 4
        assert "Long" in ls_attr.columns
        assert "TSMOM STRATEGY ASSESSMENT" in memo

    def test_strategy_returns_are_finite(self, daily_prices, config):
        results = run_backtest(daily_prices, config)
        assert np.isfinite(results["portfolio_returns"]).all()
        assert np.isfinite(results["cumulative_returns"]).all()

    def test_cumulative_stays_positive(self, daily_prices, config):
        """Cumulative equity curve should never go negative."""
        results = run_backtest(daily_prices, config)
        assert (results["cumulative_returns"] > 0).all()

    def test_weights_respect_caps(self, daily_prices, config):
        results = run_backtest(daily_prices, config)
        max_asset = config["position_sizing"]["max_asset_leverage"]
        max_port = config["position_sizing"]["max_portfolio_leverage"]

        # Per-asset
        assert (results["weights"].abs() <= max_asset + 1e-10).all().all()
        # Portfolio-level
        gross = results["weights"].abs().sum(axis=1)
        assert (gross <= max_port + 1e-10).all()

    def test_memo_rating_logic(self, config):
        """Test STRONG / MODERATE / EXPECTED rating thresholds."""
        strong_metrics = {"Sharpe": 1.5, "Max DD": -0.10, "CAGR": 0.15}
        memo = generate_memo(strong_metrics, {}, config)
        assert "STRONG" in memo

        moderate_metrics = {"Sharpe": 0.7, "Max DD": -0.30, "CAGR": 0.08}
        memo = generate_memo(moderate_metrics, {}, config)
        assert "MODERATE" in memo

        etf_metrics = {"Sharpe": 0.2, "Max DD": -0.40, "CAGR": 0.02}
        memo = generate_memo(etf_metrics, {}, config)
        assert "EXPECTED" in memo
        assert "ETF IMPLEMENTATION" in memo

    def test_format_metrics_table_is_string(self, daily_prices, config):
        results = run_backtest(daily_prices, config)
        table = build_metrics_table(results["portfolio_returns"], {}, config)
        formatted = format_metrics_table(table)
        assert isinstance(formatted, str)
        assert "Sharpe" in formatted

    def test_benchmark_cumulative_aligned(self, daily_prices, config, monkeypatch):
        mock_agg = pd.DataFrame(
            {"Close": np.linspace(100, 110, len(daily_prices))},
            index=daily_prices.index,
        )
        import tsmom.benchmarks as bm_mod
        monkeypatch.setattr(bm_mod.yf, "download", lambda *a, **kw: mock_agg)

        results = run_backtest(daily_prices, config)
        strategy_dates = results["portfolio_returns"].index
        bm_returns = build_all_benchmarks(daily_prices, strategy_dates, config)
        bm_cum = build_benchmark_cumulative(bm_returns)

        for name, cum in bm_cum.items():
            assert len(cum) > 0
            assert cum.iloc[0] > 0
