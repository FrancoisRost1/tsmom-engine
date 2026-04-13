"""
Summary tables and formatted output for console reporting.

Generates the strategy assessment memo with a 3-tier rating:
  STRONG / MODERATE / EXPECTED, ETF IMPLEMENTATION
based on Sharpe and drawdown thresholds from config.
"""

import numpy as np
import pandas as pd


def format_metrics_table(metrics_df: pd.DataFrame) -> str:
    """Format metrics comparison table for console display.

    Args:
        metrics_df: DataFrame with metrics as rows, strategy/benchmarks as columns.

    Returns:
        Formatted string table.
    """
    # Format specific rows as percentages
    pct_rows = ["CAGR", "Ann. Vol", "Max DD", "Win Rate", "Best Month", "Worst Month"]
    ratio_rows = ["Sharpe", "Sortino", "Calmar", "Skewness", "Kurtosis"]
    int_rows = ["Max DD Duration"]

    formatted = metrics_df.astype(object).copy()

    for row in pct_rows:
        if row in formatted.index:
            formatted.loc[row] = formatted.loc[row].apply(
                lambda x: f"{x:.2%}" if not np.isnan(x) else "N/A"
            )

    for row in ratio_rows:
        if row in formatted.index:
            formatted.loc[row] = formatted.loc[row].apply(
                lambda x: f"{x:.2f}" if not np.isnan(x) else "N/A"
            )

    for row in int_rows:
        if row in formatted.index:
            formatted.loc[row] = formatted.loc[row].apply(
                lambda x: f"{int(x)}m" if not np.isnan(x) else "N/A"
            )

    return formatted.to_string()


def generate_memo(
    strategy_metrics: dict,
    benchmark_metrics: dict[str, dict],
    config: dict,
) -> str:
    """Generate strategy assessment memo.

    Rating logic:
      - Sharpe >= strong_sharpe AND Max DD > max_dd_warning → STRONG
      - Sharpe >= moderate_sharpe → MODERATE
      - Otherwise → EXPECTED, ETF IMPLEMENTATION

    Args:
        strategy_metrics: Dict of TSMOM metrics.
        benchmark_metrics: Dict of benchmark name → metrics dict.
        config: Full config dict.

    Returns:
        Formatted memo string.
    """
    memo_cfg = config["memo"]
    sharpe = strategy_metrics.get("Sharpe", np.nan)
    max_dd = strategy_metrics.get("Max DD", np.nan)
    cagr = strategy_metrics.get("CAGR", np.nan)

    # Determine rating
    rating = _compute_rating(sharpe, max_dd, memo_cfg)

    lines = [
        "=" * 70,
        "TSMOM STRATEGY ASSESSMENT",
        "=" * 70,
        "",
        f"Rating: {rating}",
        "",
        "--- Key Metrics ---",
        f"  CAGR:           {cagr:.2%}" if not np.isnan(cagr) else "  CAGR:           N/A",
        f"  Sharpe Ratio:   {sharpe:.2f}" if not np.isnan(sharpe) else "  Sharpe Ratio:   N/A",
        f"  Max Drawdown:   {max_dd:.2%}" if not np.isnan(max_dd) else "  Max Drawdown:   N/A",
        "",
        "--- vs Benchmarks ---",
    ]

    # Compare vs each benchmark
    for bm_name, bm_metrics in benchmark_metrics.items():
        bm_sharpe = bm_metrics.get("Sharpe", np.nan)
        bm_cagr = bm_metrics.get("CAGR", np.nan)
        sharpe_diff = sharpe - bm_sharpe if not np.isnan(bm_sharpe) else np.nan

        if not np.isnan(sharpe_diff):
            direction = "above" if sharpe_diff > 0 else "below"
            lines.append(
                f"  vs {bm_name}: Sharpe {abs(sharpe_diff):.2f} {direction} "
                f"(TSMOM {sharpe:.2f} vs {bm_name} {bm_sharpe:.2f})"
            )

    lines.extend([
        "",
        "--- Key Findings ---",
    ])

    # What worked / didn't
    if not np.isnan(sharpe) and sharpe > 0.5:
        lines.append("  [+] Strategy delivers positive risk-adjusted returns.")
    if not np.isnan(max_dd) and max_dd > -0.20:
        lines.append("  [+] Drawdowns contained, max DD better than -20%.")
    if not np.isnan(max_dd) and max_dd < memo_cfg["max_dd_warning"]:
        lines.append(f"  [!] RISK: Max drawdown ({max_dd:.2%}) exceeds warning threshold.")
    if not np.isnan(sharpe) and sharpe < 0.5:
        lines.append("  [=] Sharpe below 0.5, consistent with ETF-based TSMOM literature.")

    # ETF implementation context (only for bottom tier)
    if rating == "EXPECTED, ETF IMPLEMENTATION":
        lines.extend([
            "",
            "--- ETF Implementation Context ---",
            "  ETF-based TSMOM structurally underperforms futures-based implementations.",
            "  Moskowitz et al. (2012) report Sharpe ~0.5-1.0 using futures across 58 markets.",
            "  ETF proxies miss roll yield, face higher implicit shorting costs, and cover a",
            "  smaller universe. A Sharpe of 0.1-0.3 is within the expected range for this",
            "  implementation choice. The value is in crisis alpha and diversification, not",
            "  standalone risk-adjusted return.",
        ])

    lines.extend([
        "",
        "--- Risk Warnings ---",
        "  - TSMOM underperforms in choppy, trendless markets (whipsaw risk).",
        "  - Short positions assume ETFs are shortable at no extra cost.",
        "  - Transaction costs modeled as flat bps, real costs may be higher.",
        "  - Past performance is not indicative of future results.",
        "",
        "=" * 70,
    ])

    return "\n".join(lines)


def _compute_rating(sharpe: float, max_dd: float, memo_cfg: dict) -> str:
    """Compute 3-tier rating for TSMOM strategy.

    STRONG:   Sharpe >= 1.0 and drawdowns contained.
    MODERATE: Sharpe >= 0.5.
    EXPECTED, ETF IMPLEMENTATION: Sharpe < 0.5 (structurally consistent
      with ETF-based TSMOM, which underperforms futures-based implementations).

    Args:
        sharpe: Strategy Sharpe ratio.
        max_dd: Max drawdown (negative).
        memo_cfg: Memo thresholds config.

    Returns:
        Rating string.
    """
    if np.isnan(sharpe):
        return "EXPECTED, ETF IMPLEMENTATION"

    if sharpe >= memo_cfg["strong_sharpe"] and max_dd > memo_cfg["max_dd_warning"]:
        return "STRONG"
    elif sharpe >= memo_cfg["moderate_sharpe"]:
        return "MODERATE"
    else:
        return "EXPECTED, ETF IMPLEMENTATION"


def print_backtest_summary(
    backtest_results: dict,
    benchmark_returns: dict[str, pd.Series],
    config: dict,
) -> None:
    """Print full backtest summary to console.

    Args:
        backtest_results: Output from run_backtest.
        benchmark_returns: Dict of benchmark returns.
        config: Full config dict.
    """
    from tsmom.analytics import compute_all_metrics, build_metrics_table

    strategy_returns = backtest_results["portfolio_returns"]

    # Build metrics table
    metrics_table = build_metrics_table(strategy_returns, benchmark_returns, config)
    print("\n" + format_metrics_table(metrics_table))

    # Generate and print memo
    strategy_metrics = compute_all_metrics(strategy_returns, config)
    benchmark_metrics_dict = {
        name: compute_all_metrics(ret, config)
        for name, ret in benchmark_returns.items()
    }
    memo = generate_memo(strategy_metrics, benchmark_metrics_dict, config)
    print("\n" + memo)
