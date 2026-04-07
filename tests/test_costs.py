"""Tests for tsmom.costs — transaction cost model."""

import numpy as np
import pandas as pd
import pytest

from tsmom.costs import compute_turnover, compute_transaction_costs


@pytest.fixture
def weights():
    dates = pd.date_range("2021-01-31", periods=4, freq="ME")
    return pd.DataFrame(
        {
            "A": [0.5, 0.3, -0.2, 0.4],
            "B": [-0.3, 0.2, 0.1, -0.1],
        },
        index=dates,
    )


class TestComputeTurnover:

    def test_first_row_is_from_zero(self, weights):
        """First rebalance turnover assumes starting from zero."""
        turnover = compute_turnover(weights)
        expected = abs(0.5) + abs(-0.3)  # |0.5-0| + |-0.3-0|
        assert abs(turnover.iloc[0] - expected) < 1e-10

    def test_subsequent_turnover(self, weights):
        turnover = compute_turnover(weights)
        # Row 1: |0.3-0.5| + |0.2-(-0.3)| = 0.2 + 0.5 = 0.7
        assert abs(turnover.iloc[1] - 0.7) < 1e-10

    def test_no_change_gives_zero_turnover(self):
        dates = pd.date_range("2021-01-31", periods=3, freq="ME")
        w = pd.DataFrame({"A": [0.5, 0.5, 0.5]}, index=dates)
        turnover = compute_turnover(w)
        # Rows 1 and 2 have zero turnover (row 0 is from zero)
        assert turnover.iloc[1] == 0.0
        assert turnover.iloc[2] == 0.0

    def test_output_length(self, weights):
        turnover = compute_turnover(weights)
        assert len(turnover) == len(weights)


class TestComputeTransactionCosts:

    def test_cost_formula(self, weights, config):
        """cost = turnover × cost_bps / 10000."""
        costs = compute_transaction_costs(weights, config)
        turnover = compute_turnover(weights)
        expected = turnover * 10 / 10000
        pd.testing.assert_series_equal(costs, expected)

    def test_costs_disabled(self, weights, config):
        config["transaction_costs"]["enabled"] = False
        costs = compute_transaction_costs(weights, config)
        assert (costs == 0).all()

    def test_costs_are_non_negative(self, weights, config):
        costs = compute_transaction_costs(weights, config)
        assert (costs >= 0).all()

    def test_higher_bps_higher_cost(self, weights, config):
        costs_10 = compute_transaction_costs(weights, config)
        config["transaction_costs"]["cost_bps"] = 50
        costs_50 = compute_transaction_costs(weights, config)
        assert (costs_50 >= costs_10).all()
