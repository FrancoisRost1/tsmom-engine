# CLAUDE.md — TSMOM Engine (Project 6)

> Local source of truth for the Time-Series Momentum Engine.
> Master CLAUDE.md in `CODE/` carries only a summary — this file has the full spec.

---

## What this project does

Cross-asset time-series momentum (TSMOM) strategy engine.
Each asset is evaluated on its **own** past returns (not relative to peers).
If an asset's trailing 12-month return (skipping the most recent month) is positive → go long.
If negative → go short.
Positions are scaled by inverse realized volatility to target a constant annualized risk per asset.

This is the canonical strategy from Moskowitz, Ooi & Pedersen (2012) "Time Series Momentum,"
implemented with ETFs across equities, bonds, commodities, and FX.

---

## Asset universe (13 ETFs via yfinance)

### Equities (4)
| Ticker | Description |
|--------|-------------|
| SPY | US Large Cap (S&P 500) |
| EFA | International Developed (MSCI EAFE) |
| EEM | Emerging Markets |
| IWM | US Small Cap (Russell 2000) |

### Bonds (3)
| Ticker | Description |
|--------|-------------|
| TLT | US Long-Term Treasuries (20yr+) |
| IEF | US Intermediate Treasuries (7-10yr) |
| LQD | US Investment Grade Corporate |

### Commodities (3)
| Ticker | Description |
|--------|-------------|
| GLD | Gold |
| DBC | Broad Commodities |
| SLV | Silver |

### FX (3)
| Ticker | Description |
|--------|-------------|
| UUP | US Dollar Index (long USD) |
| FXE | Euro (EUR/USD proxy) |
| FXY | Japanese Yen (JPY/USD proxy) |

### Design decisions on universe
- **No EMB:** mixes FX + credit risk → noisy TSMOM signal
- **No USO:** structural roll decay / contango → red flag in interviews
- **SLV added:** cleaner commodity exposure, less structural bias than oil ETFs
- **FXY added:** JPY is critical risk-off currency, adds macro regime information
- **13 total:** balanced across 4 asset classes, all liquid, all defensible

---

## Signal definition

```
signal(t) = sign( cumulative_return(t-252, t-21) )
```

- **Lookback:** 12 months (~252 trading days)
- **Skip:** most recent 1 month (~21 trading days) — avoids short-term mean reversion
- **Output:** +1 (long) or −1 (short)
- If insufficient history for an asset at time t → signal = 0 (no position)

---

## Position sizing — volatility targeting

```
raw_weight(i,t) = signal(i,t) × (target_vol / realized_vol(i,t))
```

### Realized volatility estimation (configurable)
1. **EWMA (default):** exponentially weighted, halflife = 60 days
   - `realized_vol = ewm(returns, halflife=60).std() × sqrt(252)`
2. **Simple rolling:** 63-day trailing window
   - `realized_vol = returns.rolling(63).std() × sqrt(252)`

### Position caps (both active)
- **Per-asset cap:** |raw_weight| clipped to ±2.0
- **Portfolio-level cap:** sum of |weights| clipped to 3.0 (pro-rata scale-down if exceeded)

### Cap application order
1. Compute raw weights from vol targeting
2. Clip per-asset to ±2.0
3. Compute gross leverage = sum(|weights|)
4. If gross > 3.0 → scale all weights by (3.0 / gross)

---

## Rebalancing

- **Frequency:** Monthly (last trading day of each month)
- On rebalance date:
  1. Compute 12-1 momentum signal for each asset
  2. Estimate realized vol for each asset
  3. Compute raw weights
  4. Apply position caps
  5. Deduct transaction costs on weight changes
  6. Record new portfolio weights

---

## Transaction costs

- **Configurable:** on/off toggle + adjustable bps
- **Default:** 10 bps round-trip per rebalance
- Applied as: `cost(t) = sum(|w_new - w_old|) × cost_bps / 10000`
- Deducted from portfolio return on each rebalance date

---

## Regime overlay (configurable toggle)

When `regime_overlay.enabled: true`, the strategy scales down in crisis regimes.

### Method 1 — VIX threshold (default)
- Fetch `^VIX` from yfinance
- If VIX > `vix_threshold` (default 25): scale all positions by `crisis_scale` (default 0.5)
- If VIX ≤ threshold: no adjustment

### Method 2 — HMM-based
- Fit 2-state Hidden Markov Model on SPY returns (reuse logic from Project 5)
- High-vol state → scale positions by `crisis_scale`
- Low-vol state → no adjustment

### Config
```yaml
regime_overlay:
  enabled: false          # off by default — pure TSMOM first
  method: "vix"           # "vix" or "hmm"
  vix_threshold: 25
  crisis_scale: 0.5
```

---

## Benchmarks (3)

1. **SPY** — US equity buy & hold
2. **60/40** — 60% SPY + 40% AGG, rebalanced monthly
3. **Equal-weight buy & hold** — 1/13 weight in each of the 13 ETFs, rebalanced monthly

All benchmarks computed over the same date range as the strategy.

---

## Analytics & metrics

### Portfolio-level
- CAGR, Annualized Volatility, Sharpe Ratio, Sortino Ratio
- Max Drawdown, Max Drawdown Duration
- Calmar Ratio (CAGR / Max DD)
- Skewness, Kurtosis of monthly returns
- Win rate (% of positive months)
- Best / Worst month
- All metrics computed for strategy AND each benchmark

### Attribution — asset class decomposition
- Contribution to total return by asset class (Equities, Bonds, Commodities, FX)
- Computed as: sum of (weight × return) for each asset class grouping

### Attribution — long vs short decomposition
- Separate P&L from long positions vs short positions
- Shows where TSMOM alpha actually comes from (literature says shorts are key)

### Rolling analytics
- 12-month rolling Sharpe
- 12-month rolling return (strategy vs benchmarks)
- Drawdown time series

---

## Backtest mechanics

- **Start date:** max available per ETF (ragged start — each asset enters when it has 252+ days of history)
- **End date:** most recent available data
- **Returns:** daily log returns for vol estimation, monthly simple returns for portfolio P&L
- **No lookahead bias:** all signals computed using only data available at rebalance date
- **No survivorship bias:** universe is fixed (all 13 ETFs), no dynamic selection

---

## Repo structure

```
tsmom-engine/
├── main.py                    # Orchestrator only — loads config, runs pipeline
├── config.yaml                # All parameters
├── CLAUDE.md                  # This file
├── tsmom/
│   ├── __init__.py
│   ├── loader.py              # yfinance data fetch + validation + cache
│   ├── signals.py             # 12-1 momentum signal computation
│   ├── volatility.py          # EWMA + simple rolling vol estimation
│   ├── portfolio.py           # Vol-targeted weights, position caps, rebalancing
│   ├── costs.py               # Transaction cost model
│   ├── regime.py              # VIX threshold + HMM regime overlay
│   ├── backtest.py            # Walk-forward backtest engine
│   ├── benchmarks.py          # SPY, 60/40, equal-weight benchmark construction
│   ├── analytics.py           # Performance metrics (Sharpe, DD, Calmar, etc.)
│   ├── attribution.py         # Asset class + long/short return decomposition
│   └── reporter.py            # Summary tables + formatted output
├── utils/
│   ├── __init__.py
│   └── config_loader.py       # YAML loader, returns dict
├── app/
│   └── streamlit_app.py       # Bloomberg dark mode, 5 tabs
├── data/
│   ├── raw/                   # Cached price data
│   ├── processed/             # Backtest results
│   └── cache/                 # yfinance cache
├── docs/
│   └── analysis.md            # Investment thesis + strategy write-up
├── outputs/                   # Exported reports
├── tests/
│   ├── test_signals.py
│   ├── test_volatility.py
│   ├── test_portfolio.py
│   ├── test_costs.py
│   ├── test_regime.py
│   ├── test_backtest.py
│   ├── test_benchmarks.py
│   ├── test_analytics.py
│   ├── test_attribution.py
│   └── test_integration.py
└── README.md
```

---

## Streamlit dashboard — 5 tabs

### Tab 1: OVERVIEW
- Strategy vs 3 benchmarks: cumulative return chart
- Key metrics table (CAGR, Sharpe, Max DD, Calmar) for all 4
- Current positioning: bar chart of current weights by asset
- Config summary sidebar

### Tab 2: PERFORMANCE
- Drawdown chart (strategy vs benchmarks)
- Monthly returns heatmap (year × month grid)
- Rolling 12-month Sharpe chart
- Win rate and return distribution histogram

### Tab 3: ASSET DETAIL
- Per-asset cumulative contribution chart
- Per-asset signal history (long/short timeline)
- Per-asset weight history over time
- Table: each asset's individual Sharpe, contribution, avg weight, hit rate

### Tab 4: ATTRIBUTION
- Asset class contribution stacked area chart (Equities, Bonds, Commodities, FX)
- Long vs Short P&L decomposition chart
- Long vs Short statistics table (Sharpe, return, vol for each side)
- Regime-conditional performance (if overlay enabled)

### Tab 5: ANALYSIS / MEMO
- Auto-generated strategy assessment
- Key findings (what worked, what didn't)
- Risk warnings
- Comparison vs benchmarks narrative
- Rating: STRONG / MODERATE / WEAK based on Sharpe and drawdown thresholds

---

## Key formulas reference

```python
# 12-1 momentum signal
cum_ret = price[t-21] / price[t-252] - 1
signal = +1 if cum_ret > 0 else -1

# Vol-targeted weight
realized_vol = ewm(daily_returns, halflife=60).std() * sqrt(252)
raw_weight = signal * (target_vol / realized_vol)

# Per-asset cap
weight = clip(raw_weight, -max_asset_leverage, +max_asset_leverage)

# Portfolio-level cap
gross = sum(|weights|)
if gross > max_portfolio_leverage:
    weights *= (max_portfolio_leverage / gross)

# Transaction cost
turnover = sum(|w_new - w_old|)
cost = turnover * cost_bps / 10000

# Portfolio return (monthly)
port_ret = sum(weight_i * asset_ret_i) - cost

# CAGR
cagr = (final_value / initial_value) ^ (252 / trading_days) - 1

# Sharpe
sharpe = mean(excess_returns) / std(excess_returns) * sqrt(12)  # monthly

# Max drawdown
drawdown = (cumulative_peak - cumulative_value) / cumulative_peak
max_dd = max(drawdown)
```

---

## Simplifying assumptions (document in code)

1. ETF prices used as proxies for asset class exposure (no futures, no leverage cost beyond fees)
2. Shorting via negative weight — assumes ETFs are shortable at no extra cost
3. No margin requirements modeled
4. Transaction costs are flat bps — no market impact, no bid-ask spread modeling
5. Rebalance at close prices on last trading day of month
6. VIX fetched from yfinance (`^VIX`) — may have gaps, forward-fill used
7. EWMA vol uses daily returns — not intraday
8. No cash interest earned on un-invested capital (if gross < 1.0)
9. AGG used for 60/40 benchmark bond allocation (not in TSMOM universe itself)
10. Ragged start: strategy metrics computed only from date when all assets have signals

---

## Session journal

| Date | Status | Notes |
|------|--------|-------|
| 2026-04-07 | Spec locked | Phase 1 complete. CLAUDE.md + config.yaml written. |

---

*Project CLAUDE.md — Last updated: 2026-04-07*
