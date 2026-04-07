"""
Benchmark construction: SPY buy & hold, 60/40, equal-weight.

All benchmarks computed over the same date range as the strategy
using monthly rebalancing for consistency.

Simplifying assumption: AGG used for 60/40 bond allocation (not in TSMOM universe).
"""

import pandas as pd
import yfinance as yf

from tsmom.loader import get_monthly_returns


def build_all_benchmarks(
    prices: pd.DataFrame,
    strategy_dates: pd.DatetimeIndex,
    config: dict,
) -> dict[str, pd.Series]:
    """Build all three benchmarks aligned to strategy date range.

    Args:
        prices: Daily prices for universe assets.
        strategy_dates: DatetimeIndex from the strategy's portfolio_returns.
        config: Full config dict.

    Returns:
        Dict with keys 'SPY', '60/40', 'Equal Weight', each a monthly return Series.
    """
    benchmarks = {}

    # 1. SPY buy & hold
    benchmarks["SPY"] = _build_spy_benchmark(prices, strategy_dates, config)

    # 2. 60/40 (SPY + AGG)
    benchmarks["60/40"] = _build_sixty_forty_benchmark(prices, strategy_dates, config)

    # 3. Equal-weight all 13 ETFs
    benchmarks["Equal Weight"] = _build_equal_weight_benchmark(
        prices, strategy_dates, config
    )

    # Align all benchmarks to a common date range so metric comparisons
    # are apples-to-apples (same months for every series).
    common_idx = strategy_dates
    for series in benchmarks.values():
        common_idx = common_idx.intersection(series.index)
    benchmarks = {name: s.loc[common_idx] for name, s in benchmarks.items()}

    return benchmarks


def _build_spy_benchmark(
    prices: pd.DataFrame,
    strategy_dates: pd.DatetimeIndex,
    config: dict,
) -> pd.Series:
    """SPY buy & hold monthly returns.

    Args:
        prices: Daily prices (must contain 'SPY' column).
        strategy_dates: Dates to align to.
        config: Config dict.

    Returns:
        Monthly return Series for SPY.
    """
    monthly_ret = get_monthly_returns(prices[["SPY"]])["SPY"]
    return monthly_ret.reindex(strategy_dates).dropna()


def _build_sixty_forty_benchmark(
    prices: pd.DataFrame,
    strategy_dates: pd.DatetimeIndex,
    config: dict,
) -> pd.Series:
    """60% SPY + 40% AGG, rebalanced monthly.

    Downloads AGG if not in the price data.

    Args:
        prices: Daily prices.
        strategy_dates: Dates to align to.
        config: Config dict.

    Returns:
        Monthly return Series for 60/40 portfolio.
    """
    bench_cfg = config["benchmarks"]["sixty_forty"]
    eq_weight = bench_cfg["equity_weight"]
    bond_weight = bench_cfg["bond_weight"]
    bond_ticker = bench_cfg["bond_ticker"]

    # Get SPY monthly returns
    spy_monthly = get_monthly_returns(prices[["SPY"]])["SPY"]

    # Fetch AGG separately (not in TSMOM universe)
    agg_data = yf.download(bond_ticker, period="max", auto_adjust=True, progress=False)
    if isinstance(agg_data.columns, pd.MultiIndex):
        agg_prices = agg_data["Close"].squeeze()
    else:
        agg_prices = agg_data["Close"]
    agg_prices = agg_prices.to_frame(bond_ticker)
    agg_monthly = get_monthly_returns(agg_prices)[bond_ticker]

    # Combine
    combined = pd.DataFrame({"SPY": spy_monthly, bond_ticker: agg_monthly}).dropna()
    sixty_forty = combined["SPY"] * eq_weight + combined[bond_ticker] * bond_weight
    sixty_forty.name = "60/40"

    return sixty_forty.reindex(strategy_dates).dropna()


def _build_equal_weight_benchmark(
    prices: pd.DataFrame,
    strategy_dates: pd.DatetimeIndex,
    config: dict,
) -> pd.Series:
    """Equal-weight (1/N) all 13 ETFs, rebalanced monthly.

    Args:
        prices: Daily prices for all universe assets.
        strategy_dates: Dates to align to.
        config: Config dict.

    Returns:
        Monthly return Series for equal-weight portfolio.
    """
    monthly_ret = get_monthly_returns(prices)

    # Only include months where all assets have data so the basket is always
    # a true 1/N of the full universe, not a varying subset.
    complete = monthly_ret.dropna()
    eq_weight = complete.mean(axis=1)
    eq_weight.name = "Equal Weight"

    return eq_weight.reindex(strategy_dates).dropna()


def build_benchmark_cumulative(
    benchmark_returns: dict[str, pd.Series],
) -> dict[str, pd.Series]:
    """Build cumulative return series for each benchmark.

    Args:
        benchmark_returns: Dict of benchmark name → monthly return Series.

    Returns:
        Dict of benchmark name → cumulative value Series (starts at 1.0).
    """
    cumulative = {}
    for name, returns in benchmark_returns.items():
        cum = (1 + returns).cumprod()
        cum.name = name
        cumulative[name] = cum
    return cumulative
