"""
Performance metrics: Sharpe, Sortino, CAGR, max drawdown, Calmar, etc.

All metrics computed for strategy AND each benchmark.
Uses monthly returns with annualization via sqrt(12).
"""

import numpy as np
import pandas as pd


def compute_all_metrics(monthly_returns: pd.Series, config: dict) -> dict:
    """Compute full suite of performance metrics from monthly returns.

    Args:
        monthly_returns: Monthly return Series.
        config: Full config dict (uses analytics.risk_free_rate).

    Returns:
        Dict of metric name → value.
    """
    rf = config["analytics"]["risk_free_rate"]
    rf_monthly = (1 + rf) ** (1 / 12) - 1

    cumulative = (1 + monthly_returns).cumprod()
    n_months = len(monthly_returns)
    n_years = n_months / 12

    metrics = {}

    # CAGR
    if n_years > 0 and cumulative.iloc[-1] > 0:
        metrics["CAGR"] = (cumulative.iloc[-1]) ** (1 / n_years) - 1
    else:
        metrics["CAGR"] = np.nan

    # Annualized volatility
    metrics["Ann. Vol"] = monthly_returns.std() * np.sqrt(12)

    # Sharpe ratio
    excess = monthly_returns - rf_monthly
    if monthly_returns.std() > 0:
        metrics["Sharpe"] = excess.mean() / monthly_returns.std() * np.sqrt(12)
    else:
        metrics["Sharpe"] = np.nan

    # Sortino ratio (downside deviation)
    # Industry standard: DD = sqrt(mean(min(r - rf, 0)^2)) over ALL periods,
    # not just negative ones.  Including zero shortfalls from positive months
    # in the denominator is required; omitting them shrinks N and biases the ratio.
    downside_diff = np.minimum(excess.values, 0)
    downside_std = np.sqrt(np.mean(downside_diff**2)) if len(excess) > 0 else np.nan
    if downside_std and downside_std > 0:
        metrics["Sortino"] = excess.mean() / downside_std * np.sqrt(12)
    else:
        metrics["Sortino"] = np.nan

    # Max drawdown
    dd_series = compute_drawdown_series(monthly_returns)
    metrics["Max DD"] = dd_series.min()

    # Max drawdown duration (in months)
    metrics["Max DD Duration"] = _max_dd_duration(dd_series)

    # Calmar ratio
    if metrics["Max DD"] != 0 and not np.isnan(metrics["Max DD"]):
        metrics["Calmar"] = metrics["CAGR"] / abs(metrics["Max DD"])
    else:
        metrics["Calmar"] = np.nan

    # Skewness and kurtosis
    metrics["Skewness"] = monthly_returns.skew()
    metrics["Kurtosis"] = monthly_returns.kurtosis()

    # Win rate
    metrics["Win Rate"] = (monthly_returns > 0).mean()

    # Best / worst month
    metrics["Best Month"] = monthly_returns.max()
    metrics["Worst Month"] = monthly_returns.min()

    return metrics


def compute_drawdown_series(monthly_returns: pd.Series) -> pd.Series:
    """Compute drawdown series from monthly returns.

    drawdown(t) = (peak(t) - value(t)) / peak(t)
    Returned as negative values (e.g., -0.15 = 15% drawdown).

    Args:
        monthly_returns: Monthly return Series.

    Returns:
        Drawdown Series (negative values = drawdowns).
    """
    cumulative = (1 + monthly_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown


def _max_dd_duration(dd_series: pd.Series) -> int:
    """Compute max drawdown duration in months.

    Duration = longest streak where drawdown < 0.

    Args:
        dd_series: Drawdown Series.

    Returns:
        Max drawdown duration in months.
    """
    in_drawdown = dd_series < 0
    if not in_drawdown.any():
        return 0

    # Group consecutive drawdown periods
    groups = (in_drawdown != in_drawdown.shift()).cumsum()
    dd_groups = groups[in_drawdown]
    if dd_groups.empty:
        return 0

    return dd_groups.value_counts().max()


def compute_rolling_sharpe(
    monthly_returns: pd.Series, config: dict
) -> pd.Series:
    """Compute rolling 12-month Sharpe ratio.

    Args:
        monthly_returns: Monthly return Series.
        config: Full config dict.

    Returns:
        Rolling Sharpe Series.
    """
    window = config["analytics"]["rolling_window_months"]
    rf = config["analytics"]["risk_free_rate"]
    rf_monthly = (1 + rf) ** (1 / 12) - 1

    excess = monthly_returns - rf_monthly
    rolling_mean = excess.rolling(window).mean()
    rolling_std = monthly_returns.rolling(window).std()

    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(12)
    rolling_sharpe.name = "Rolling Sharpe"

    return rolling_sharpe


def compute_rolling_return(
    monthly_returns: pd.Series, config: dict
) -> pd.Series:
    """Compute rolling 12-month cumulative return.

    Args:
        monthly_returns: Monthly return Series.
        config: Full config dict.

    Returns:
        Rolling 12-month return Series.
    """
    window = config["analytics"]["rolling_window_months"]

    rolling_ret = (1 + monthly_returns).rolling(window).apply(np.prod, raw=True) - 1
    rolling_ret.name = "Rolling 12M Return"

    return rolling_ret


def build_metrics_table(
    strategy_returns: pd.Series,
    benchmark_returns: dict[str, pd.Series],
    config: dict,
) -> pd.DataFrame:
    """Build comparison table of metrics for strategy + all benchmarks.

    Args:
        strategy_returns: Monthly return Series for TSMOM.
        benchmark_returns: Dict of benchmark name → monthly return Series.
        config: Full config dict.

    Returns:
        DataFrame with metrics as rows, strategy/benchmarks as columns.
    """
    all_series = {"TSMOM": strategy_returns, **benchmark_returns}

    metrics_dict = {}
    for name, returns in all_series.items():
        metrics_dict[name] = compute_all_metrics(returns, config)

    table = pd.DataFrame(metrics_dict)
    return table
