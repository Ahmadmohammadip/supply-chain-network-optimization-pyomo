"""Phase 2: the strategic layer — which facilities open, and why.

The headline is the fixed-versus-variable trade-off from PROJECT_BRIEF.md
section 4: a cheap-to-open, expensive-to-ship warehouse should lose to an
expensive-to-open, cheap-to-ship one once volume is large enough. That crossover
is the clearest evidence the model is doing something non-trivial rather than
just routing flow.
"""

import pytest

from scn_opt.model.builder import build_scn_model
from scn_opt.solve import solve_scn


def _solve(case, **overrides):
    return solve_scn(build_scn_model(**{**case, **overrides}))


def _tradeoff_case(demand_per_period, n_periods=4):
    """Two warehouses, identical but for how they trade fixed against variable.

    Costs are stripped to nothing except the two levers, so the crossover can be
    computed by hand:

        cheap to open:  100 + 5 x (demand x periods)
        cheap to ship: 1000 + 1 x (demand x periods)

    Equal when 100 + 20D = 1000 + 4D, i.e. D = 56.25 over four periods. Below
    that the low fixed cost wins; above it the low shipping cost does.
    """
    return dict(
        plants=["P1"],
        warehouses=["W_cheap_fixed", "W_cheap_shipping"],
        customers=["C1"],
        n_periods=n_periods,
        demand={("C1", t): float(demand_per_period) for t in range(1, n_periods + 1)},
        plant_capacity={"P1": 10_000.0},
        production_cost={"P1": 0.0},
        throughput_capacity={"W_cheap_fixed": 10_000.0, "W_cheap_shipping": 10_000.0},
        storage_capacity={"W_cheap_fixed": 10_000.0, "W_cheap_shipping": 10_000.0},
        holding_cost={"W_cheap_fixed": 0.0, "W_cheap_shipping": 0.0},
        safety_stock={"W_cheap_fixed": 0.0, "W_cheap_shipping": 0.0},
        initial_inventory={"W_cheap_fixed": 0.0, "W_cheap_shipping": 0.0},
        cost_plant_to_warehouse={
            ("P1", "W_cheap_fixed"): 0.0,
            ("P1", "W_cheap_shipping"): 0.0,
        },
        cost_warehouse_to_customer={
            ("W_cheap_fixed", "C1"): 5.0,
            ("W_cheap_shipping", "C1"): 1.0,
        },
        fixed_cost_plant={"P1": 0.0},
        fixed_cost_warehouse={"W_cheap_fixed": 100.0, "W_cheap_shipping": 1000.0},
    )


def test_low_volume_favours_the_cheap_to_open_warehouse():
    result = _solve(_tradeoff_case(demand_per_period=20))

    assert result.open_warehouses == ["W_cheap_fixed"]
    assert result.total_cost == pytest.approx(100 + 20 * 20, abs=1e-6)  # 500


def test_high_volume_favours_the_cheap_to_ship_warehouse():
    result = _solve(_tradeoff_case(demand_per_period=100))

    assert result.open_warehouses == ["W_cheap_shipping"]
    assert result.total_cost == pytest.approx(1000 + 4 * 100, abs=1e-6)  # 1400


def test_the_crossover_lands_where_the_arithmetic_says():
    # 100 + 20D = 1000 + 4D gives D = 56.25, so the decision must flip between
    # 56 and 57 units per period and nowhere else.
    below = _solve(_tradeoff_case(demand_per_period=56))
    above = _solve(_tradeoff_case(demand_per_period=57))

    assert below.open_warehouses == ["W_cheap_fixed"]
    assert below.total_cost == pytest.approx(100 + 20 * 56, abs=1e-6)  # 1220

    assert above.open_warehouses == ["W_cheap_shipping"]
    assert above.total_cost == pytest.approx(1000 + 4 * 57, abs=1e-6)  # 1228


def test_a_longer_horizon_shifts_the_same_decision():
    # Fixed cost is charged once for the whole horizon, so lengthening it
    # accumulates variable cost against an unchanged one-off. The same demand
    # that favours the cheap-to-open site over four periods should favour the
    # cheap-to-ship one over long enough a horizon.
    short = _solve(_tradeoff_case(demand_per_period=20, n_periods=4))
    long = _solve(_tradeoff_case(demand_per_period=20, n_periods=40))

    assert short.open_warehouses == ["W_cheap_fixed"]
    assert long.open_warehouses == ["W_cheap_shipping"]


def test_unused_facilities_stay_closed(case_factory):
    # A second warehouse that costs money to open and offers nothing should not
    # be opened.
    case = case_factory(
        warehouses=["W_useful", "W_pointless"],
        throughput_capacity={"W_useful": 100.0, "W_pointless": 100.0},
        storage_capacity={"W_useful": 100.0, "W_pointless": 100.0},
        holding_cost={"W_useful": 0.5, "W_pointless": 0.5},
        safety_stock={"W_useful": 0.0, "W_pointless": 0.0},
        initial_inventory={"W_useful": 0.0, "W_pointless": 0.0},
        cost_plant_to_warehouse={("P1", "W_useful"): 2.0, ("P1", "W_pointless"): 2.0},
        cost_warehouse_to_customer={("W_useful", "C1"): 3.0, ("W_pointless", "C1"): 3.0},
        fixed_cost_warehouse={"W_useful": 10.0, "W_pointless": 500.0},
    )
    result = _solve(case)

    assert result.open_warehouses == ["W_useful"]
    assert result.fixed_cost == pytest.approx(10.0, abs=1e-6)


def test_no_flow_leaves_a_closed_facility(case_factory):
    case = case_factory(
        warehouses=["W_useful", "W_pointless"],
        throughput_capacity={"W_useful": 100.0, "W_pointless": 100.0},
        storage_capacity={"W_useful": 100.0, "W_pointless": 100.0},
        holding_cost={"W_useful": 0.5, "W_pointless": 0.5},
        safety_stock={"W_useful": 0.0, "W_pointless": 0.0},
        initial_inventory={"W_useful": 0.0, "W_pointless": 0.0},
        cost_plant_to_warehouse={("P1", "W_useful"): 2.0, ("P1", "W_pointless"): 2.0},
        cost_warehouse_to_customer={("W_useful", "C1"): 3.0, ("W_pointless", "C1"): 3.0},
        fixed_cost_warehouse={"W_useful": 10.0, "W_pointless": 500.0},
    )
    result = _solve(case)

    closed = set(case["warehouses"]) - set(result.open_warehouses)
    for w in closed:
        assert not any(key[1] == w for key in result.flows_plant_to_warehouse)
        assert not any(key[0] == w for key in result.flows_warehouse_to_customer)
        for t in (1, 2, 3):
            assert result.inventory[(w, t)] == pytest.approx(0.0, abs=1e-6)


def test_opening_everything_never_beats_choosing(case_factory):
    # The open-everything plan is a feasible point of the optimized problem, so
    # optimizing can never do worse. If this ever fails, the binaries are wrong.
    case = case_factory(
        warehouses=["W_useful", "W_pointless"],
        throughput_capacity={"W_useful": 100.0, "W_pointless": 100.0},
        storage_capacity={"W_useful": 100.0, "W_pointless": 100.0},
        holding_cost={"W_useful": 0.5, "W_pointless": 0.5},
        safety_stock={"W_useful": 0.0, "W_pointless": 0.0},
        initial_inventory={"W_useful": 0.0, "W_pointless": 0.0},
        cost_plant_to_warehouse={("P1", "W_useful"): 2.0, ("P1", "W_pointless"): 2.0},
        cost_warehouse_to_customer={("W_useful", "C1"): 3.0, ("W_pointless", "C1"): 3.0},
        fixed_cost_warehouse={"W_useful": 10.0, "W_pointless": 500.0},
    )

    optimized = _solve(case)
    everything = _solve(case, force_open_all=True)

    assert everything.total_cost >= optimized.total_cost - 1e-6
    assert len(everything.open_warehouses) == 2
    # Here the difference is exactly the pointless warehouse's fixed cost.
    assert everything.total_cost - optimized.total_cost == pytest.approx(500.0, abs=1e-6)


def test_force_open_all_still_charges_fixed_costs(case_factory):
    # The counterfactual has to include fixed costs, otherwise it would not be
    # comparable with the optimized objective.
    case = case_factory(fixed_cost_warehouse={"W1": 42.0}, fixed_cost_plant={"P1": 8.0})
    result = _solve(case, force_open_all=True)

    assert result.fixed_cost == pytest.approx(50.0, abs=1e-6)
    assert sum(result.cost_breakdown().values()) == pytest.approx(
        result.total_cost, abs=1e-6
    )


def test_a_warehouse_holding_opening_stock_is_forced_open(case_factory):
    # There is no disposal variable, so inventory cannot be written off: a
    # warehouse that starts with stock cannot be closed, whatever it costs.
    # Setting initial_inventory therefore pre-decides that facility.
    case = case_factory(
        warehouses=["W_stocked", "W_cheap"],
        throughput_capacity={"W_stocked": 100.0, "W_cheap": 100.0},
        storage_capacity={"W_stocked": 100.0, "W_cheap": 100.0},
        holding_cost={"W_stocked": 0.5, "W_cheap": 0.5},
        safety_stock={"W_stocked": 0.0, "W_cheap": 0.0},
        initial_inventory={"W_stocked": 15.0, "W_cheap": 0.0},
        cost_plant_to_warehouse={("P1", "W_stocked"): 2.0, ("P1", "W_cheap"): 2.0},
        cost_warehouse_to_customer={("W_stocked", "C1"): 3.0, ("W_cheap", "C1"): 3.0},
        # Ruinously expensive to open, yet it opens anyway.
        fixed_cost_warehouse={"W_stocked": 9999.0, "W_cheap": 1.0},
    )
    result = _solve(case)

    assert "W_stocked" in result.open_warehouses


def test_cost_breakdown_sums_to_the_objective(case_factory):
    case = case_factory(
        demand={("C1", 1): 30.0, ("C1", 2): 10.0, ("C1", 3): 45.0},
        plant_capacity={"P1": 40.0},
        safety_stock={"W1": 5.0},
        fixed_cost_plant={"P1": 120.0},
        fixed_cost_warehouse={"W1": 75.0},
    )
    result = _solve(case)

    breakdown = result.cost_breakdown()
    assert sum(breakdown.values()) == pytest.approx(result.total_cost, abs=1e-6)
    assert breakdown["fixed"] == pytest.approx(195.0, abs=1e-6)
    assert breakdown["holding"] > 0  # safety stock has to be carried
