"""Tests for tsmom.portfolio — position sizing and caps."""

import numpy as np
import pandas as pd
import pytest

from tsmom.portfolio import (
    compute_raw_weights,
    apply_position_caps,
    build_weight_history,
    compute_gross_leverage,
    compute_net_exposure,
)


@pytest.fixture
def signals():
    dates = pd.date_range("2021-01-31", periods=4, freq="ME")
    return pd.DataFrame(
        {"A": [1, -1, 1, 0], "B": [-1, 1, 1, -1], "C": [1, 1, -1, 1]},
        index=dates,
    )


@pytest.fixture
def realized_vol():
    dates = pd.date_range("2021-01-31", periods=4, freq="ME")
    return pd.DataFrame(
        {"A": [0.20, 0.15, 0.10, 0.25], "B": [0.10, 0.20, 0.30, 0.05], "C": [0.15, 0.10, 0.20, 0.15]},
        index=dates,
    )


class TestComputeRawWeights:

    def test_formula_correct(self, signals, realized_vol, config):
        """raw_weight = signal * (target_vol / realized_vol)."""
        raw = compute_raw_weights(signals, realized_vol, config)
        target = config["position_sizing"]["target_vol"]  # 0.10
        # Row 0, asset A: 1 * (0.10 / 0.20) = 0.5
        assert abs(raw.iloc[0]["A"] - 0.5) < 1e-10
        # Row 1, asset B: 1 * (0.10 / 0.20) = 0.5
        assert abs(raw.iloc[1]["B"] - 0.5) < 1e-10

    def test_zero_signal_gives_zero_weight(self, signals, realized_vol, config):
        raw = compute_raw_weights(signals, realized_vol, config)
        # Row 3, asset A has signal 0
        assert raw.iloc[3]["A"] == 0.0

    def test_nan_vol_gives_zero_weight(self, signals, config):
        vol = signals.copy().astype(float)
        vol.iloc[:] = np.nan
        raw = compute_raw_weights(signals, vol, config)
        assert (raw == 0).all().all()

    def test_zero_vol_gives_zero_weight(self, signals, config):
        vol = signals.copy().astype(float)
        vol.iloc[:] = 0.0
        raw = compute_raw_weights(signals, vol, config)
        assert (raw == 0).all().all()

    def test_negative_signal_gives_negative_weight(self, signals, realized_vol, config):
        raw = compute_raw_weights(signals, realized_vol, config)
        # Row 0, asset B: signal = -1 → weight < 0
        assert raw.iloc[0]["B"] < 0


class TestApplyPositionCaps:

    def test_per_asset_cap(self, config):
        """Weights exceeding ±2.0 should be clipped."""
        dates = pd.date_range("2021-01-31", periods=1, freq="ME")
        weights = pd.DataFrame({"A": [5.0], "B": [-3.0]}, index=dates)
        capped = apply_position_caps(weights, config)
        assert capped.iloc[0]["A"] <= 2.0
        assert capped.iloc[0]["B"] >= -2.0

    def test_portfolio_level_cap(self, config):
        """Gross leverage must not exceed max_portfolio_leverage (3.0)."""
        dates = pd.date_range("2021-01-31", periods=1, freq="ME")
        weights = pd.DataFrame(
            {"A": [2.0], "B": [2.0], "C": [2.0]}, index=dates
        )
        capped = apply_position_caps(weights, config)
        gross = capped.abs().sum(axis=1).iloc[0]
        assert gross <= config["position_sizing"]["max_portfolio_leverage"] + 1e-10

    def test_pro_rata_scaling(self, config):
        """When gross exceeds cap, all weights scale proportionally."""
        dates = pd.date_range("2021-01-31", periods=1, freq="ME")
        weights = pd.DataFrame(
            {"A": [1.5], "B": [1.5], "C": [1.5]}, index=dates
        )
        # Gross = 4.5 > 3.0, expect scale by 3.0/4.5 = 2/3
        capped = apply_position_caps(weights, config)
        expected = 1.5 * (3.0 / 4.5)
        assert abs(capped.iloc[0]["A"] - expected) < 1e-10

    def test_small_weights_pass_through(self, config):
        """Weights within both caps should not be altered."""
        dates = pd.date_range("2021-01-31", periods=1, freq="ME")
        weights = pd.DataFrame(
            {"A": [0.3], "B": [-0.2]}, index=dates
        )
        capped = apply_position_caps(weights, config)
        pd.testing.assert_frame_equal(capped, weights)


class TestBuildWeightHistory:

    def test_end_to_end(self, signals, realized_vol, config):
        weights = build_weight_history(signals, realized_vol, config)
        # Output shape matches input
        assert weights.shape == signals.shape
        # Gross leverage never exceeds cap
        gross = weights.abs().sum(axis=1)
        assert (gross <= config["position_sizing"]["max_portfolio_leverage"] + 1e-10).all()


class TestLeverageHelpers:

    def test_gross_leverage(self, sample_weights):
        gross = compute_gross_leverage(sample_weights)
        expected_first = abs(0.5) + abs(-0.2) + abs(0.3) + abs(0.1) + abs(-0.1)
        assert abs(gross.iloc[0] - expected_first) < 1e-10

    def test_net_exposure(self, sample_weights):
        net = compute_net_exposure(sample_weights)
        expected_first = 0.5 + (-0.2) + 0.3 + 0.1 + (-0.1)
        assert abs(net.iloc[0] - expected_first) < 1e-10
