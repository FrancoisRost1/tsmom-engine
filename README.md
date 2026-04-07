# TSMOM Engine

Cross-asset time-series momentum strategy engine implementing Moskowitz, Ooi & Pedersen (2012). Each asset is evaluated on its own past returns (not relative to peers). Positions are volatility-targeted and rebalanced monthly across equities, bonds, commodities, and FX.

## Key Results

| Metric | TSMOM | SPY | 60/40 | Equal Weight |
|--------|-------|-----|-------|--------------|
| CAGR | 5.87% | 10.35% | 7.66% | 5.36% |
| Ann. Vol | 15.68% | 15.42% | 9.88% | 8.93% |
| Sharpe | 0.19 | 0.46 | 0.40 | 0.19 |
| Max DD | -44.75% | -50.78% | -32.32% | -21.88% |
| Win Rate | 55.03% | 65.94% | 66.38% | 58.52% |

Backtest period: 1993-03 to 2026-04 (398 months, 13 ETFs).

## Features

- **12-1 momentum signal** -- sign of cumulative return from t-252 to t-21 trading days
- **Volatility targeting** -- EWMA (halflife 60d) or rolling vol, 10% annualized target per asset
- **Position caps** -- per-asset (2x) and portfolio-level (3x gross) with pro-rata scaling
- **Transaction cost model** -- configurable flat bps on turnover (default 10 bps)
- **Regime overlay** -- optional VIX threshold or expanding-window HMM (no lookahead bias)
- **3 benchmarks** -- SPY buy & hold, 60/40 (SPY + AGG), equal-weight universe
- **Full attribution** -- asset class decomposition + long/short P&L split
- **103 unit tests** -- all synthetic data, zero external API calls
- **5-tab Streamlit dashboard** -- Bloomberg dark mode with Plotly charts
- **Codex-audited** -- reviewed by GPT-5.4 as senior quant analyst; 7 findings fixed

## Asset Universe (13 ETFs)

| Class | Tickers | Description |
|-------|---------|-------------|
| Equities | SPY, EFA, EEM, IWM | US Large/Small Cap, Intl Developed, EM |
| Bonds | TLT, IEF, LQD | Long/Intermediate Treasuries, IG Corporate |
| Commodities | GLD, DBC, SLV | Gold, Broad Commodities, Silver |
| FX | UUP, FXE, FXY | USD Index, Euro, Yen |

No EMB (noisy mixed signal), no USO (structural contango/roll decay).

## Project Structure

```
tsmom-engine/
├── main.py                    # Orchestrator
├── config.yaml                # All parameters (zero hardcoded numbers)
├── tsmom/
│   ├── loader.py              # yfinance fetch + cache + validation
│   ├── signals.py             # 12-1 momentum signal
│   ├── volatility.py          # EWMA + rolling vol estimation
│   ├── portfolio.py           # Vol-targeted weights + position caps
│   ├── costs.py               # Transaction cost model
│   ├── regime.py              # VIX threshold + HMM overlay
│   ├── backtest.py            # Walk-forward backtest engine
│   ├── benchmarks.py          # SPY, 60/40, equal-weight
│   ├── analytics.py           # Sharpe, Sortino, CAGR, drawdown, Calmar
│   ├── attribution.py         # Asset class + long/short decomposition
│   └── reporter.py            # Console output + strategy memo
├── app/
│   └── streamlit_app.py       # 5-tab Bloomberg dark mode dashboard
├── tests/                     # 103 tests (pytest)
├── docs/
│   └── analysis.md            # Investment thesis + strategy write-up
└── data/
    ├── cache/                 # yfinance price cache
    └── processed/             # Backtest outputs (CSV)
```

## How to Run

### Install dependencies

```bash
pip install pandas numpy yfinance pyyaml streamlit plotly pytest scipy
```

### Run the backtest

```bash
python3 main.py
```

This fetches price data, runs the full TSMOM pipeline, prints metrics, and saves results to `data/processed/`.

### Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

### Run tests

```bash
pytest tests/ -v
```

## Dashboard

5 tabs: Overview, Performance, Asset Detail, Attribution, Analysis/Memo.

![Dashboard Screenshot](docs/dashboard_screenshot.png)
*Screenshot placeholder -- run the dashboard locally to see the full Bloomberg dark mode interface.*

## Methodology

**Signal:** `sign(price[t-21] / price[t-252] - 1)` -- positive trailing 11-month return = long, negative = short.

**Position sizing:** `weight = signal * (target_vol / realized_vol)`, clipped to +/-2x per asset, 3x gross portfolio.

**Rebalancing:** Monthly (last trading day). Transaction costs deducted on turnover.

**Reference:** Moskowitz, T., Ooi, Y. H., & Pedersen, L. H. (2012). Time Series Momentum. *Journal of Financial Economics*, 104(2), 228-250.

## Simplifying Assumptions

1. ETF prices as proxies (no futures, no leverage cost)
2. Shorting via negative weight at no extra cost
3. No margin requirements
4. Flat bps transaction costs (no market impact)
5. Rebalance at close on last trading day of month
6. EWMA vol from daily returns (not intraday)
7. No cash interest on un-invested capital
8. Fixed ETF universe selected ex-post (inception bias acknowledged)

## License

MIT
