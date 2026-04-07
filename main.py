"""
TSMOM Engine — Orchestrator.

Loads config, fetches data, runs backtest, builds benchmarks, and prints results.
This file orchestrates only — all logic lives in the tsmom/ package.

Cross-asset time-series momentum strategy from Moskowitz, Ooi & Pedersen (2012).
"""

from utils.config_loader import load_config, get_all_tickers
from tsmom.loader import fetch_prices, validate_prices, compute_returns
from tsmom.backtest import run_backtest, get_strategy_date_range
from tsmom.benchmarks import build_all_benchmarks, build_benchmark_cumulative
from tsmom.analytics import build_metrics_table
from tsmom.reporter import print_backtest_summary


def main():
    """Run the full TSMOM pipeline."""
    # 1. Load config
    print("Loading configuration...")
    config = load_config("config.yaml")

    # 2. Fetch and validate price data
    tickers = get_all_tickers(config)
    print(f"Fetching price data for {len(tickers)} assets: {', '.join(tickers)}")
    prices = fetch_prices(tickers, config)
    prices = validate_prices(prices, tickers)
    print(f"Price data: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")

    # 3. Run backtest
    print("Running TSMOM backtest...")
    results = run_backtest(prices, config)

    start, end = get_strategy_date_range(results)
    print(f"Strategy period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    print(f"Months: {len(results['portfolio_returns'])}")

    # 4. Build benchmarks
    print("Building benchmarks...")
    strategy_dates = results["portfolio_returns"].index
    benchmark_returns = build_all_benchmarks(prices, strategy_dates, config)
    benchmark_cumulative = build_benchmark_cumulative(benchmark_returns)

    # 5. Print results
    print_backtest_summary(results, benchmark_returns, config)

    # 6. Save processed data
    _save_results(results, benchmark_returns, benchmark_cumulative, config)

    print("\nDone. Results saved to data/processed/")


def _save_results(
    results: dict,
    benchmark_returns: dict,
    benchmark_cumulative: dict,
    config: dict,
) -> None:
    """Save backtest results to CSV for Streamlit consumption.

    Args:
        results: Backtest results dict.
        benchmark_returns: Dict of benchmark monthly returns.
        benchmark_cumulative: Dict of benchmark cumulative returns.
        config: Full config dict.
    """
    import pandas as pd
    from pathlib import Path

    out_dir = Path(config["paths"]["processed_data"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Strategy returns
    results["portfolio_returns"].to_csv(out_dir / "strategy_returns.csv")
    results["cumulative_returns"].to_csv(out_dir / "strategy_cumulative.csv")

    # Weights (raw rebalance-date weights + return-aligned shifted weights)
    results["weights"].to_csv(out_dir / "weights.csv")
    results["weights_aligned"].to_csv(out_dir / "weights_aligned.csv")

    # Signals
    results["signals"].to_csv(out_dir / "signals.csv")

    # Costs
    results["costs"].to_csv(out_dir / "costs.csv")

    # Benchmark returns
    bm_df = pd.DataFrame(benchmark_returns)
    bm_df.to_csv(out_dir / "benchmark_returns.csv")

    # Benchmark cumulative
    bm_cum_df = pd.DataFrame(benchmark_cumulative)
    bm_cum_df.to_csv(out_dir / "benchmark_cumulative.csv")


if __name__ == "__main__":
    main()
