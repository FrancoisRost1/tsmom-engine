"""Tests for tsmom.regime — regime overlay."""

import numpy as np
import pandas as pd
import pytest

from tsmom.regime import apply_regime_overlay, get_regime_labels


@pytest.fixture
def weights():
    dates = pd.date_range("2021-01-31", periods=3, freq="ME")
    return pd.DataFrame(
        {"SPY": [0.5, 0.6, 0.4], "TLT": [0.3, -0.2, 0.1]}, index=dates
    )


@pytest.fixture
def prices():
    dates = pd.bdate_range("2020-01-02", periods=400, freq="B")
    np.random.seed(42)
    return pd.DataFrame(
        {"SPY": 100 + np.cumsum(np.random.normal(0.05, 1, 400)),
         "TLT": 50 + np.cumsum(np.random.normal(0.02, 0.5, 400))},
        index=dates,
    )


class TestApplyRegimeOverlay:

    def test_disabled_returns_unchanged(self, weights, prices, config):
        """When regime_overlay.enabled = false, weights are unchanged."""
        rebal = weights.index
        result = apply_regime_overlay(weights, prices, rebal, config)
        pd.testing.assert_frame_equal(result, weights)

    def test_vix_overlay_scales_down(self, weights, prices, config, monkeypatch):
        """When VIX > threshold, all weights should be scaled by crisis_scale."""
        config["regime_overlay"]["enabled"] = True
        config["regime_overlay"]["method"] = "vix"
        rebal = weights.index

        # Mock yf.download to return high VIX
        vix_dates = pd.bdate_range("2020-01-02", periods=400, freq="B")
        mock_vix = pd.DataFrame(
            {"Close": [30.0] * 400}, index=vix_dates
        )

        import tsmom.regime as regime_mod
        monkeypatch.setattr(regime_mod.yf, "download", lambda *a, **kw: mock_vix)

        result = apply_regime_overlay(weights, prices, rebal, config)
        expected = weights * config["regime_overlay"]["crisis_scale"]
        pd.testing.assert_frame_equal(result, expected)

    def test_vix_overlay_no_scale_below_threshold(self, weights, prices, config, monkeypatch):
        """When VIX <= threshold, weights should be unchanged."""
        config["regime_overlay"]["enabled"] = True
        config["regime_overlay"]["method"] = "vix"
        rebal = weights.index

        vix_dates = pd.bdate_range("2020-01-02", periods=400, freq="B")
        mock_vix = pd.DataFrame(
            {"Close": [15.0] * 400}, index=vix_dates
        )

        import tsmom.regime as regime_mod
        monkeypatch.setattr(regime_mod.yf, "download", lambda *a, **kw: mock_vix)

        result = apply_regime_overlay(weights, prices, rebal, config)
        pd.testing.assert_frame_equal(result, weights)

    def test_invalid_method_raises(self, weights, prices, config):
        config["regime_overlay"]["enabled"] = True
        config["regime_overlay"]["method"] = "invalid"
        with pytest.raises(ValueError, match="Unknown regime method"):
            apply_regime_overlay(weights, prices, weights.index, config)


class TestGetRegimeLabels:

    def test_disabled_returns_all_normal(self, prices, config):
        dates = pd.date_range("2021-01-31", periods=3, freq="ME")
        labels = get_regime_labels(prices, dates, config)
        assert (labels == "normal").all()

    def test_vix_labels(self, prices, config, monkeypatch):
        config["regime_overlay"]["enabled"] = True
        dates = pd.date_range("2021-01-31", periods=3, freq="ME")

        vix_dates = pd.bdate_range("2020-01-02", periods=400, freq="B")
        mock_vix = pd.DataFrame(
            {"Close": [30.0] * 400}, index=vix_dates
        )

        import tsmom.regime as regime_mod
        monkeypatch.setattr(regime_mod.yf, "download", lambda *a, **kw: mock_vix)

        labels = get_regime_labels(prices, dates, config)
        assert (labels == "crisis").all()
