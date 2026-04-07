# TSMOM Engine -- Investment Analysis

## Investment Thesis

Time-series momentum (TSMOM) exploits the empirical tendency of assets to continue trending in their recent direction. Unlike cross-sectional momentum (buying winners, selling losers relative to peers), TSMOM trades each asset against its own history: if an asset's trailing 12-month return (skipping the most recent month) is positive, go long; if negative, go short.

The academic foundation is strong. Moskowitz, Ooi & Pedersen (2012) document significant TSMOM profits across 58 futures markets spanning equities, bonds, currencies, and commodities over multiple decades. The strategy is particularly valuable because:

1. **Crisis alpha.** TSMOM historically profits during extended market declines (2008, 2020) by capturing sustained downtrends via short positions.
2. **Diversification.** Cross-asset implementation reduces dependence on any single market regime.
3. **Volatility targeting.** Position sizing inversely proportional to realized vol creates a self-stabilizing mechanism.

The thesis here is that a disciplined, rules-based TSMOM implementation using liquid ETFs should deliver positive risk-adjusted returns uncorrelated with traditional buy-and-hold.

## Strategy Assessment

**Rating: WEAK** (Sharpe 0.19, Max DD -44.75%)

This vanilla ETF-based TSMOM underperforms its benchmarks. The key results:

| Metric | TSMOM | SPY | 60/40 | Equal Weight |
|--------|-------|-----|-------|--------------|
| CAGR | 5.87% | 10.35% | 7.66% | 5.36% |
| Sharpe | 0.19 | 0.46 | 0.40 | 0.19 |
| Sortino | 0.27 | 0.67 | 0.57 | 0.27 |
| Max DD | -44.75% | -50.78% | -32.32% | -21.88% |
| Calmar | 0.13 | 0.20 | 0.24 | 0.24 |
| Win Rate | 55.03% | 65.94% | 66.38% | 58.52% |

## What Worked

1. **Drawdown management vs pure equity.** TSMOM max drawdown (-44.75%) is better than SPY (-50.78%), demonstrating the crisis alpha property. During sustained sell-offs, the strategy correctly goes short, partially offsetting losses.

2. **Consistent signal generation.** The 12-1 momentum signal produces meaningful positioning -- the strategy is not flat most of the time. 55% win rate confirms modest positive edge.

3. **Volatility targeting mechanics.** The EWMA vol targeting + position caps work as designed. Gross leverage stays within bounds and the portfolio doesn't blow up during vol spikes.

4. **Cross-asset diversification.** The 4-asset-class, 13-ETF universe distributes risk. No single asset class dominates the P&L.

## What Didn't Work

1. **Whipsaw drag.** Monthly rebalancing in trendless (choppy) markets generates frequent signal flips, each incurring transaction costs for negative expected return. This is the primary Sharpe killer.

2. **Short-side performance.** In a secular equity bull market (2009-2024), short equity positions are a persistent drag. The "crisis alpha" appears only intermittently.

3. **ETF vs. futures gap.** The original Moskowitz et al. results use futures, which have zero financing cost for the notional and can capture roll yield. ETF-based implementation misses these effects, particularly in commodities (DBC has structural decay) and FX.

4. **Inception bias.** Many ETFs in the universe launched mid-2000s (DBC: 2006, SLV: 2006, UUP: 2007). The strategy's "full sample" is really only 13-asset complete since ~2007. Earlier periods trade a partial universe with less diversification.

5. **Flat position sizing.** Using a single target vol (10%) for all assets ignores that some assets have stronger trend persistence than others. Bonds and FX tend to trend more reliably than equity indices.

## Key Risks

1. **Regime risk.** TSMOM performs poorly in mean-reverting or range-bound markets. A prolonged period without sustained trends (like parts of 2015-2016) erodes capital through whipsaw.

2. **Crowding.** Trend-following is now a well-known strategy. When many CTAs and quant funds run similar signals, crowded exits during regime shifts can amplify drawdowns.

3. **Implementation gap.** ETFs have tracking error, management fees, and potential liquidity issues during stress. These are not captured in the flat 10 bps cost model.

4. **Leverage constraints.** The strategy can reach 3x gross leverage. In practice, this requires a prime brokerage relationship and margin that is not modeled here.

5. **Left-tail risk.** Despite vol targeting, the strategy has negative skewness (-0.49) and excess kurtosis (1.89), meaning extreme losses are more frequent than a normal distribution would suggest.

## Return Scenarios

### Bull Case (Sharpe > 0.5)
- **Trigger:** Sustained directional trends across multiple asset classes (e.g., rising rates + falling equities + commodity supercycle).
- **Expected CAGR:** 8-12%
- **Historical precedent:** 2007-2009 financial crisis; 2020 COVID crash and recovery; 2022 rate hiking cycle.
- **What's needed:** Enable regime overlay (VIX or HMM) to reduce whipsaw in calm markets, concentrate risk when trends are strong.

### Base Case (Sharpe 0.1-0.3)
- **Trigger:** Mixed regimes -- some trending periods offset by choppy intervals. This is essentially what the backtest shows.
- **Expected CAGR:** 4-7%
- **Conclusion:** Vanilla TSMOM on ETFs with monthly rebalancing is a marginal strategy on its own. It may add value as a diversifier in a multi-strategy portfolio but does not justify standalone allocation.

### Bear Case (Sharpe < 0)
- **Trigger:** Extended mean-reversion regime. Rapid V-shaped recoveries (like March-April 2020) whipsaw the signal before it can adjust. Or a period of very low volatility where the vol-targeting inflates position sizes right before a regime break.
- **Expected CAGR:** -2% to +2%
- **Risk:** Max drawdown could exceed -50% if multiple asset classes whipsaw simultaneously.

## Potential Improvements

1. **Faster signal variants.** Test 3-month and 6-month lookbacks alongside 12-month. Blend signals across horizons.
2. **Asymmetric volatility targeting.** Scale up when trends are strong (e.g., positive momentum AND rising vol in direction of trend), scale down in choppy markets.
3. **Regime-conditional sizing.** The VIX overlay (currently disabled) or HMM could reduce positions in crisis regimes where whipsaw risk is highest.
4. **Futures migration.** Replace ETFs with futures for commodity and FX exposure. Eliminates tracking error and captures roll yield.
5. **Strategy combination.** TSMOM works best as one component in a multi-strategy portfolio (paired with carry, value, or defensive strategies).

## Conclusion

This implementation correctly captures the mechanics of Moskowitz et al. (2012) TSMOM using ETFs. The strategy demonstrates real crisis alpha (better max DD than SPY) and proper risk management via vol targeting and position caps. However, in a 30+ year backtest dominated by equity bull markets, the whipsaw cost of monthly signal flips outweighs the intermittent trend-following profits.

The honest assessment: vanilla TSMOM on ETFs with monthly rebalancing is not competitive as a standalone strategy. Its value lies in (a) demonstrating the implementation rigor required for systematic macro, and (b) serving as a building block for more sophisticated multi-strategy frameworks where its crisis alpha property provides genuine diversification.

---

*Analysis written: 2026-04-07*
*Backtest period: 1993-03 to 2026-04 (398 months)*
*Reference: Moskowitz, T., Ooi, Y. H., & Pedersen, L. H. (2012). Time Series Momentum. Journal of Financial Economics, 104(2), 228-250.*
