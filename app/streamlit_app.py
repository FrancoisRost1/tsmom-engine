"""
TSMOM Engine, Bloomberg Dark Mode Dashboard.

5-tab Streamlit app: Overview, Performance, Asset Detail, Attribution, Analysis/Memo.
Reads pre-computed CSV data from data/processed/ (run main.py first).
"""

import sys
from pathlib import Path

# Add project root to path so imports work when running from app/ directory
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from style_inject import inject_styles, styled_header, styled_kpi, styled_divider, TOKENS, apply_plotly_theme
from utils.config_loader import load_config, get_all_tickers, get_asset_class_map
from tsmom.analytics import compute_all_metrics, compute_drawdown_series, compute_rolling_sharpe, compute_rolling_return
from tsmom.attribution import (
    compute_asset_class_attribution,
    compute_cumulative_asset_class_attribution,
    compute_long_short_attribution,
    compute_long_short_statistics,
    compute_per_asset_attribution,
)
from tsmom.reporter import _compute_rating

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG & STYLES
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TSMOM Engine",
    page_icon="</>",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()

# ─────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────
DATA_DIR = ROOT / "data" / "processed"


@st.cache_data
def load_data():
    """Load all pre-computed backtest results."""
    config = load_config(str(ROOT / "config.yaml"))

    strat_ret = pd.read_csv(DATA_DIR / "strategy_returns.csv", index_col=0, parse_dates=True).squeeze()
    strat_cum = pd.read_csv(DATA_DIR / "strategy_cumulative.csv", index_col=0, parse_dates=True).squeeze()
    weights = pd.read_csv(DATA_DIR / "weights.csv", index_col=0, parse_dates=True)
    weights_aligned = pd.read_csv(DATA_DIR / "weights_aligned.csv", index_col=0, parse_dates=True)
    signals = pd.read_csv(DATA_DIR / "signals.csv", index_col=0, parse_dates=True)
    costs = pd.read_csv(DATA_DIR / "costs.csv", index_col=0, parse_dates=True).squeeze()
    bm_ret = pd.read_csv(DATA_DIR / "benchmark_returns.csv", index_col=0, parse_dates=True)
    bm_cum = pd.read_csv(DATA_DIR / "benchmark_cumulative.csv", index_col=0, parse_dates=True)

    return config, strat_ret, strat_cum, weights, weights_aligned, signals, costs, bm_ret, bm_cum


config, strat_ret, strat_cum, weights, weights_aligned, signals, costs, bm_ret, bm_cum = load_data()

# Derived data
tickers = list(weights.columns)
ac_map = get_asset_class_map(config)
strat_metrics = compute_all_metrics(strat_ret, config)
bm_metrics = {col: compute_all_metrics(bm_ret[col], config) for col in bm_ret.columns}

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    c_primary = TOKENS["text_primary"]
    c_muted = TOKENS["text_muted"]
    st.markdown(f"<h2 style='color:{c_primary};margin-bottom:0;'>TSMOM Engine</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{c_muted};font-size:0.8rem;margin-top:0.2rem;'>Time-Series Momentum Strategy</p>", unsafe_allow_html=True)
    styled_divider()

    st.markdown(f"**Strategy period**")
    st.markdown(f"`{strat_ret.index[0].strftime('%Y-%m-%d')}` to `{strat_ret.index[-1].strftime('%Y-%m-%d')}`")
    st.markdown(f"**Months:** {len(strat_ret)}")
    st.markdown(f"**Assets:** {len(tickers)}")

    styled_divider()
    st.markdown("**Config**")
    st.markdown(f"Signal: {config['signal']['lookback_days']}d lookback, {config['signal']['skip_days']}d skip")
    st.markdown(f"Vol method: {config['volatility']['method']} (halflife {config['volatility']['ewma_halflife']})")
    st.markdown(f"Target vol: {config['position_sizing']['target_vol']:.0%}")
    st.markdown(f"Max asset lev: {config['position_sizing']['max_asset_leverage']}x")
    st.markdown(f"Max port lev: {config['position_sizing']['max_portfolio_leverage']}x")
    st.markdown(f"Costs: {'ON' if config['transaction_costs']['enabled'] else 'OFF'} ({config['transaction_costs']['cost_bps']} bps)")
    st.markdown(f"Regime overlay: {'ON' if config['regime_overlay']['enabled'] else 'OFF'}")

# ─────────────────────────────────────────────────────────────────
# HELPER: Plotly defaults
# ─────────────────────────────────────────────────────────────────
COLORS = {
    "TSMOM": TOKENS["accent_primary"],
    "SPY": TOKENS["accent_warning"],
    "60/40": TOKENS["accent_success"],
    "Equal Weight": TOKENS["accent_info"],
}


def _fmt_pct(v):
    return f"{v:.2%}" if not np.isnan(v) else "N/A"


def _fmt_ratio(v):
    return f"{v:.2f}" if not np.isnan(v) else "N/A"


def _delta_color(v):
    if np.isnan(v):
        return TOKENS["text_muted"]
    return TOKENS["accent_success"] if v >= 0 else TOKENS["accent_danger"]


# ─────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────
styled_header("TSMOM Engine", "Cross-asset time-series momentum | Moskowitz, Ooi & Pedersen (2012)")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["OVERVIEW", "PERFORMANCE", "ASSET DETAIL", "ATTRIBUTION", "ANALYSIS"])

# ═════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═════════════════════════════════════════════════════════════════
with tab1:
    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        styled_kpi("CAGR", _fmt_pct(strat_metrics["CAGR"]),
                    delta=f"vs SPY {_fmt_pct(strat_metrics['CAGR'] - bm_metrics['SPY']['CAGR'])}",
                    delta_color=_delta_color(strat_metrics["CAGR"] - bm_metrics["SPY"]["CAGR"]))
    with k2:
        styled_kpi("Sharpe", _fmt_ratio(strat_metrics["Sharpe"]),
                    delta=f"vs SPY {_fmt_ratio(strat_metrics['Sharpe'] - bm_metrics['SPY']['Sharpe'])}",
                    delta_color=_delta_color(strat_metrics["Sharpe"] - bm_metrics["SPY"]["Sharpe"]))
    with k3:
        styled_kpi("Max Drawdown", _fmt_pct(strat_metrics["Max DD"]),
                    delta_color=TOKENS["accent_danger"])
    with k4:
        styled_kpi("Calmar", _fmt_ratio(strat_metrics["Calmar"]),
                    delta=f"vs SPY {_fmt_ratio(strat_metrics['Calmar'] - bm_metrics['SPY']['Calmar'])}",
                    delta_color=_delta_color(strat_metrics["Calmar"] - bm_metrics["SPY"]["Calmar"]))

    st.markdown("")

    # Cumulative return chart
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=strat_cum.index, y=strat_cum.values, name="TSMOM",
                                  line=dict(color=COLORS["TSMOM"], width=2.5)))
    for col in bm_cum.columns:
        fig_cum.add_trace(go.Scatter(x=bm_cum.index, y=bm_cum[col].values, name=col,
                                      line=dict(color=COLORS.get(col, TOKENS["text_muted"]), width=1.5, dash="dot")))
    fig_cum.update_layout(title="Cumulative Returns (Growth of $1)", yaxis_title="Cumulative Value",
                           hovermode="x unified", legend=dict(orientation="h", y=-0.15))
    apply_plotly_theme(fig_cum)
    st.plotly_chart(fig_cum, use_container_width=True)

    # Metrics comparison table + Current weights
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### Key Metrics Comparison")
        rows = ["CAGR", "Ann. Vol", "Sharpe", "Sortino", "Max DD", "Calmar", "Win Rate"]
        table_data = {"Metric": rows}
        table_data["TSMOM"] = [_fmt_pct(strat_metrics[r]) if r in ["CAGR", "Ann. Vol", "Max DD", "Win Rate"]
                                else _fmt_ratio(strat_metrics[r]) for r in rows]
        for bm_name, bm_m in bm_metrics.items():
            table_data[bm_name] = [_fmt_pct(bm_m[r]) if r in ["CAGR", "Ann. Vol", "Max DD", "Win Rate"]
                                    else _fmt_ratio(bm_m[r]) for r in rows]
        st.dataframe(pd.DataFrame(table_data).set_index("Metric"), use_container_width=True)

    with col_right:
        st.markdown("#### Current Positioning")
        last_weights = weights.iloc[-1].sort_values()
        colors_bar = [TOKENS["accent_danger"] if v < 0 else TOKENS["accent_success"] for v in last_weights.values]
        fig_pos = go.Figure(go.Bar(
            x=last_weights.values, y=last_weights.index, orientation="h",
            marker_color=colors_bar, text=[f"{v:.2f}" for v in last_weights.values],
            textposition="outside",
        ))
        fig_pos.update_layout(title=f"Weights ({weights.index[-1].strftime('%b %Y')})",
                               xaxis_title="Weight", yaxis_title="", height=400)
        apply_plotly_theme(fig_pos)
        st.plotly_chart(fig_pos, use_container_width=True)

# ═════════════════════════════════════════════════════════════════
# TAB 2: PERFORMANCE
# ═════════════════════════════════════════════════════════════════
with tab2:
    # Drawdown chart
    dd_strat = compute_drawdown_series(strat_ret)
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dd_strat.index, y=dd_strat.values, name="TSMOM",
                                 fill="tozeroy", line=dict(color=COLORS["TSMOM"], width=1.5),
                                 fillcolor="rgba(99,102,241,0.15)"))
    for col in bm_ret.columns:
        dd_bm = compute_drawdown_series(bm_ret[col])
        fig_dd.add_trace(go.Scatter(x=dd_bm.index, y=dd_bm.values, name=col,
                                     line=dict(color=COLORS.get(col, TOKENS["text_muted"]), width=1, dash="dot")))
    fig_dd.update_layout(title="Underwater Chart (Drawdowns)", yaxis_title="Drawdown",
                          yaxis_tickformat=".0%", hovermode="x unified",
                          legend=dict(orientation="h", y=-0.15))
    apply_plotly_theme(fig_dd)
    st.plotly_chart(fig_dd, use_container_width=True)

    # Monthly heatmap + Rolling Sharpe side by side
    col_heat, col_roll = st.columns([3, 2])

    with col_heat:
        st.markdown("#### Monthly Returns Heatmap")
        monthly_df = strat_ret.copy()
        monthly_df.index = pd.to_datetime(monthly_df.index)
        heat_df = pd.DataFrame({
            "Year": monthly_df.index.year,
            "Month": monthly_df.index.month,
            "Return": monthly_df.values,
        })
        heat_pivot = heat_df.pivot_table(index="Year", columns="Month", values="Return", aggfunc="first")
        heat_pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        fig_heat = go.Figure(data=go.Heatmap(
            z=heat_pivot.values,
            x=heat_pivot.columns,
            y=heat_pivot.index,
            colorscale=[[0, TOKENS["accent_danger"]], [0.5, TOKENS["bg_surface"]], [1, TOKENS["accent_success"]]],
            zmid=0,
            text=[[f"{v:.1%}" if not np.isnan(v) else "" for v in row] for row in heat_pivot.values],
            texttemplate="%{text}",
            textfont=dict(size=10),
            hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2%}<extra></extra>",
            colorbar=dict(tickformat=".0%", title="Return"),
        ))
        fig_heat.update_layout(title="", height=450, yaxis=dict(autorange="reversed"))
        apply_plotly_theme(fig_heat)
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_roll:
        st.markdown("#### Rolling 12M Sharpe")
        roll_sharpe = compute_rolling_sharpe(strat_ret, config)
        fig_rs = go.Figure()
        fig_rs.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe.values, name="TSMOM",
                                     line=dict(color=COLORS["TSMOM"], width=2)))
        fig_rs.add_hline(y=0, line_dash="dash", line_color=TOKENS["text_muted"], line_width=0.5)
        fig_rs.add_hline(y=1.0, line_dash="dash", line_color=TOKENS["accent_success"], line_width=0.5,
                          annotation_text="Sharpe=1.0", annotation_position="top right")
        fig_rs.update_layout(title="", yaxis_title="Sharpe Ratio", hovermode="x unified", height=450)
        apply_plotly_theme(fig_rs)
        st.plotly_chart(fig_rs, use_container_width=True)

    # Return distribution
    st.markdown("#### Return Distribution")
    col_hist, col_stats = st.columns([3, 1])
    with col_hist:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=strat_ret.values, nbinsx=40, name="TSMOM",
            marker_color=TOKENS["accent_primary"], opacity=0.75,
        ))
        fig_hist.add_vline(x=0, line_dash="dash", line_color=TOKENS["text_muted"])
        fig_hist.add_vline(x=strat_ret.mean(), line_dash="dash", line_color=TOKENS["accent_success"],
                            annotation_text=f"Mean: {strat_ret.mean():.2%}")
        fig_hist.update_layout(title="", xaxis_title="Monthly Return", yaxis_title="Count",
                                xaxis_tickformat=".1%", bargap=0.05)
        apply_plotly_theme(fig_hist)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_stats:
        st.markdown("")
        st.markdown("")
        styled_kpi("Win Rate", _fmt_pct(strat_metrics["Win Rate"]))
        st.markdown("")
        styled_kpi("Skewness", _fmt_ratio(strat_metrics["Skewness"]))
        st.markdown("")
        styled_kpi("Kurtosis", _fmt_ratio(strat_metrics["Kurtosis"]))

# ═════════════════════════════════════════════════════════════════
# TAB 3: ASSET DETAIL
# ═════════════════════════════════════════════════════════════════
with tab3:
    # Per-asset cumulative contribution
    # Use pre-aligned weights and returns from backtest (no manual shifting needed)

    # Load monthly asset returns aligned to strategy period.
    # Pre-computed and shipped in data/processed/ so the dashboard works on
    # Streamlit Cloud without the gitignored yfinance cache.
    @st.cache_data
    def get_monthly_asset_returns():
        return pd.read_csv(
            DATA_DIR / "monthly_asset_returns.csv",
            index_col=0,
            parse_dates=True,
        )

    monthly_asset_ret = get_monthly_asset_returns()

    per_asset_cum = compute_per_asset_attribution(weights_aligned, monthly_asset_ret)

    st.markdown("#### Per-Asset Cumulative Contribution")
    fig_pa = go.Figure()
    for i, col in enumerate(per_asset_cum.columns):
        fig_pa.add_trace(go.Scatter(
            x=per_asset_cum.index, y=per_asset_cum[col].values, name=col,
            line=dict(width=1.5),
        ))
    fig_pa.update_layout(title="", yaxis_title="Cumulative Contribution", hovermode="x unified",
                          legend=dict(orientation="h", y=-0.2), yaxis_tickformat=".2f")
    apply_plotly_theme(fig_pa)
    st.plotly_chart(fig_pa, use_container_width=True)

    # Signal history + Weight history
    col_sig, col_wt = st.columns(2)

    with col_sig:
        st.markdown("#### Signal History (Long / Short)")
        fig_sig = go.Figure()
        for i, col in enumerate(signals.columns):
            fig_sig.add_trace(go.Scatter(
                x=signals.index, y=signals[col].values + i * 2.5,
                name=col, mode="lines", line=dict(width=1.5),
                hovertemplate=f"{col}: %{{text}}<extra></extra>",
                text=["Long" if v == 1 else "Short" if v == -1 else "Flat" for v in signals[col].values],
            ))
        fig_sig.update_layout(title="", yaxis=dict(showticklabels=False), hovermode="x unified",
                               legend=dict(orientation="h", y=-0.15), height=500)
        apply_plotly_theme(fig_sig)
        st.plotly_chart(fig_sig, use_container_width=True)

    with col_wt:
        st.markdown("#### Weight History Over Time")
        fig_wh = go.Figure()
        for col in weights.columns:
            fig_wh.add_trace(go.Scatter(
                x=weights.index, y=weights[col].values, name=col,
                line=dict(width=1.5), stackgroup=None,
            ))
        fig_wh.update_layout(title="", yaxis_title="Weight", hovermode="x unified",
                              legend=dict(orientation="h", y=-0.15), height=500)
        apply_plotly_theme(fig_wh)
        st.plotly_chart(fig_wh, use_container_width=True)

    # Per-asset stats table
    st.markdown("#### Per-Asset Statistics")
    asset_stats = []
    for ticker in tickers:
        if ticker not in monthly_asset_ret.columns or ticker not in weights_aligned.columns:
            continue
        common = weights_aligned.index.intersection(monthly_asset_ret.index)
        w_t = weights_aligned[ticker].loc[common]
        r_t = monthly_asset_ret[ticker].loc[common]
        contrib = (w_t * r_t)
        total_contrib = contrib.sum()
        avg_weight = w_t.mean()
        hit_rate = (contrib > 0).mean()
        # Individual contribution Sharpe
        if contrib.std() > 0:
            sharpe_c = contrib.mean() / contrib.std() * np.sqrt(12)
        else:
            sharpe_c = np.nan
        asset_stats.append({
            "Asset": ticker,
            "Class": ac_map.get(ticker, ""),
            "Total Contrib.": f"{total_contrib:.4f}",
            "Avg Weight": f"{avg_weight:.3f}",
            "Hit Rate": f"{hit_rate:.1%}",
            "Contrib. Sharpe": f"{sharpe_c:.2f}" if not np.isnan(sharpe_c) else "N/A",
        })

    st.dataframe(pd.DataFrame(asset_stats).set_index("Asset"), use_container_width=True)

# ═════════════════════════════════════════════════════════════════
# TAB 4: ATTRIBUTION
# ═════════════════════════════════════════════════════════════════
with tab4:
    # Asset class contribution stacked area
    cum_ac = compute_cumulative_asset_class_attribution(weights_aligned, monthly_asset_ret, config)

    st.markdown("#### Asset Class Contribution (Cumulative)")
    fig_ac = go.Figure()
    ac_colors = {
        "equities": TOKENS["accent_primary"],
        "bonds": TOKENS["accent_success"],
        "commodities": TOKENS["accent_warning"],
        "fx": TOKENS["accent_info"],
    }
    for ac in cum_ac.columns:
        fig_ac.add_trace(go.Scatter(
            x=cum_ac.index, y=cum_ac[ac].values, name=ac.title(),
            stackgroup="one", line=dict(width=0.5, color=ac_colors.get(ac, TOKENS["text_muted"])),
            fillcolor=ac_colors.get(ac, TOKENS["text_muted"]),
        ))
    fig_ac.update_layout(title="", yaxis_title="Cumulative Contribution", hovermode="x unified",
                          legend=dict(orientation="h", y=-0.15), yaxis_tickformat=".2f")
    apply_plotly_theme(fig_ac)
    st.plotly_chart(fig_ac, use_container_width=True)

    # Long vs Short decomposition
    ls_attr = compute_long_short_attribution(weights_aligned, monthly_asset_ret)
    ls_cum = ls_attr.cumsum()

    col_ls_chart, col_ls_table = st.columns([3, 2])

    with col_ls_chart:
        st.markdown("#### Long vs Short P&L Decomposition")
        fig_ls = go.Figure()
        fig_ls.add_trace(go.Scatter(
            x=ls_cum.index, y=ls_cum["Long"].values, name="Long",
            line=dict(color=TOKENS["accent_success"], width=2),
        ))
        fig_ls.add_trace(go.Scatter(
            x=ls_cum.index, y=ls_cum["Short"].values, name="Short",
            line=dict(color=TOKENS["accent_danger"], width=2),
        ))
        fig_ls.add_trace(go.Scatter(
            x=ls_cum.index, y=(ls_cum["Long"] + ls_cum["Short"]).values, name="Total",
            line=dict(color=TOKENS["text_primary"], width=1.5, dash="dot"),
        ))
        fig_ls.update_layout(title="", yaxis_title="Cumulative P&L", hovermode="x unified",
                              legend=dict(orientation="h", y=-0.15), yaxis_tickformat=".2f")
        apply_plotly_theme(fig_ls)
        st.plotly_chart(fig_ls, use_container_width=True)

    with col_ls_table:
        st.markdown("#### Long vs Short Statistics")
        ls_stats = compute_long_short_statistics(ls_attr, config)
        # Format for display, cast to object dtype first so pandas >=2.2
        # allows writing formatted strings back into what were numeric rows
        # (otherwise it raises LossySetitemError on the row assignment).
        display_stats = ls_stats.astype(object)
        for row in ["Ann. Return", "Ann. Vol", "Avg Monthly"]:
            if row in display_stats.index:
                display_stats.loc[row] = ls_stats.loc[row].apply(lambda x: f"{x:.4f}")
        for row in ["Sharpe"]:
            if row in display_stats.index:
                display_stats.loc[row] = ls_stats.loc[row].apply(lambda x: f"{x:.2f}" if not np.isnan(x) else "N/A")
        for row in ["Hit Rate"]:
            if row in display_stats.index:
                display_stats.loc[row] = ls_stats.loc[row].apply(lambda x: f"{x:.1%}")
        st.dataframe(display_stats, use_container_width=True)

    # Regime-conditional performance (if enabled)
    if config["regime_overlay"]["enabled"]:
        st.markdown("#### Regime-Conditional Performance")
        st.info("Regime overlay is enabled: split performance by Normal vs Crisis periods.")
    else:
        st.markdown("#### Regime Overlay")
        st.info("Regime overlay is disabled. Enable in config.yaml to see regime-conditional performance.")

# ═════════════════════════════════════════════════════════════════
# TAB 5: ANALYSIS / MEMO
# ═════════════════════════════════════════════════════════════════
with tab5:
    # Rating badge
    rating = _compute_rating(strat_metrics["Sharpe"], strat_metrics["Max DD"], config["memo"])
    rating_colors = {
        "STRONG": TOKENS["accent_success"],
        "MODERATE": TOKENS["accent_warning"],
        "EXPECTED: ETF IMPLEMENTATION": TOKENS["accent_info"],
    }
    rc = rating_colors.get(rating, TOKENS["text_muted"])

    st.markdown(
        f"<div style='text-align:center;margin-bottom:1.5rem;'>"
        f"<span style='background:{rc};color:#fff;padding:0.5rem 2rem;border-radius:8px;"
        f"font-size:1.5rem;font-weight:700;letter-spacing:0.05em;'>{rating}</span></div>",
        unsafe_allow_html=True,
    )

    # Key metrics recap
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        styled_kpi("CAGR", _fmt_pct(strat_metrics["CAGR"]))
    with k2:
        styled_kpi("Sharpe", _fmt_ratio(strat_metrics["Sharpe"]))
    with k3:
        styled_kpi("Max Drawdown", _fmt_pct(strat_metrics["Max DD"]))
    with k4:
        styled_kpi("Sortino", _fmt_ratio(strat_metrics["Sortino"]))

    styled_divider()

    # Benchmark comparison narrative
    st.markdown("#### vs Benchmarks")
    for bm_name, bm_m in bm_metrics.items():
        sharpe_diff = strat_metrics["Sharpe"] - bm_m["Sharpe"]
        cagr_diff = strat_metrics["CAGR"] - bm_m["CAGR"]
        direction = "above" if sharpe_diff > 0 else "below"
        icon = "+" if sharpe_diff > 0 else ""
        st.markdown(
            f"- **vs {bm_name}:** Sharpe {abs(sharpe_diff):.2f} {direction} "
            f"(TSMOM {strat_metrics['Sharpe']:.2f} vs {bm_name} {bm_m['Sharpe']:.2f}) | "
            f"CAGR gap: {icon}{cagr_diff:.2%}"
        )

    styled_divider()

    # Key findings
    st.markdown("#### Key Findings")
    findings = []

    if strat_metrics["Sharpe"] > 0.5:
        findings.append(("success", "Strategy delivers positive risk-adjusted returns (Sharpe > 0.5)."))
    elif strat_metrics["Sharpe"] > 0:
        findings.append(("info", f"Sharpe {strat_metrics['Sharpe']:.2f}: consistent with ETF-based TSMOM. "
                          "Futures implementations typically achieve 0.5-1.0 (Moskowitz et al. 2012)."))
    else:
        findings.append(("warning", f"Negative risk-adjusted returns (Sharpe {strat_metrics['Sharpe']:.2f})."))

    if strat_metrics["Max DD"] > -0.20:
        findings.append(("success", f"Drawdowns well contained (Max DD {strat_metrics['Max DD']:.2%})."))
    elif strat_metrics["Max DD"] > -0.30:
        findings.append(("warning", f"Moderate drawdown risk (Max DD {strat_metrics['Max DD']:.2%})."))
    else:
        findings.append(("warning", f"Significant drawdown risk (Max DD {strat_metrics['Max DD']:.2%}), "
                          "though better than SPY buy & hold during major crises."))

    if strat_metrics["Win Rate"] > 0.55:
        findings.append(("success", f"Positive hit rate ({strat_metrics['Win Rate']:.1%} of months profitable)."))

    # Long vs short insight
    ls_total = ls_attr.sum()
    if ls_total["Short"] > 0:
        findings.append(("success", "Short positions contribute positively: consistent with TSMOM literature."))
    else:
        findings.append(("info", "Short positions are a drag: expected in secular bull markets. "
                          "The equity risk premium creates a structural headwind for short ETF positions. "
                          "See the Attribution tab for the full long/short decomposition."))

    for level, msg in findings:
        if level == "success":
            st.success(msg)
        elif level == "warning":
            st.warning(msg)
        elif level == "info":
            st.info(msg)
        else:
            st.error(msg)

    styled_divider()

    # Risk warnings
    st.markdown("#### Risk Warnings")
    st.markdown("""
- TSMOM underperforms in choppy, trendless markets (whipsaw risk).
- Short positions assume ETFs are shortable at no extra cost.
- Transaction costs modeled as flat bps: real costs may be higher in stress.
- ETF proxies do not capture futures roll yield or basis effects.
- Past performance is not indicative of future results.
    """)

    styled_divider()

    # Methodology
    with st.expander("Methodology"):
        st.markdown(f"""
**Signal:** 12-1 momentum: `sign(price[t-21] / price[t-252] - 1)`. Positive trailing return = long, negative = short.

**Position sizing:** Volatility targeting: each asset scaled to {config['position_sizing']['target_vol']:.0%} annualized vol.
Per-asset cap: {config['position_sizing']['max_asset_leverage']}x. Portfolio gross cap: {config['position_sizing']['max_portfolio_leverage']}x.

**Rebalancing:** Monthly (last trading day). Transaction costs: {config['transaction_costs']['cost_bps']} bps round-trip.

**Universe:** {len(tickers)} ETFs across equities, bonds, commodities, and FX.

**Reference:** Moskowitz, Ooi & Pedersen (2012), "Time Series Momentum," *Journal of Financial Economics*.
        """)
