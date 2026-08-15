"""Shared test setup and hand-checkable instances.

The Agg backend is selected here, before anything imports pyplot, so the
plotting tests run headless in CI the same way they do locally.

`make_case` builds the primitive keyword arguments `build_scn_model` takes, with
a base instance small enough to verify on paper. Overriding any single argument
gives a focused variant, which keeps each test's deviation from the baseline
visible at a glance.

Everything shared lives on fixtures rather than module constants, so test files
never import from conftest (which pytest's default import mode does not put on
the path).
"""

import matplotlib
import pytest

matplotlib.use("Agg")


def make_case(**overrides) -> dict:
    """Keyword arguments for `build_scn_model`.

    Base case: one plant, one warehouse, one customer, three periods, demand 10
    per period. Shipping costs 2 inbound and 3 outbound, production costs 1, and
    holding costs 0.5 — so holding is never worth it and the optimum ships
    exactly to demand:

        production 30x1 + inbound 30x2 + outbound 30x3 + holding 0 = 180
    """
    case = dict(
        plants=["P1"],
        warehouses=["W1"],
        customers=["C1"],
        n_periods=3,
        demand={("C1", 1): 10.0, ("C1", 2): 10.0, ("C1", 3): 10.0},
        plant_capacity={"P1": 100.0},
        production_cost={"P1": 1.0},
        throughput_capacity={"W1": 100.0},
        storage_capacity={"W1": 100.0},
        holding_cost={"W1": 0.5},
        safety_stock={"W1": 0.0},
        initial_inventory={"W1": 0.0},
        cost_plant_to_warehouse={("P1", "W1"): 2.0},
        cost_warehouse_to_customer={("W1", "C1"): 3.0},
    )
    case.update(overrides)
    return case


BASE_CASE_OPTIMAL_COST = 180.0


@pytest.fixture
def case():
    """The base hand-checkable case as a kwargs dict."""
    return make_case()


@pytest.fixture
def case_factory():
    """`make_case` itself, for tests that need several variants."""
    return make_case


@pytest.fixture
def base_case_cost():
    return BASE_CASE_OPTIMAL_COST
