"""Phase 1: the inventory recursion, the safety-stock floor, and the case that
justifies the model being multi-period at all."""

import pytest

from scn_opt.model.builder import build_scn_model
from scn_opt.solve import solve_scn


def _solve(case):
    return solve_scn(build_scn_model(**case))


def _inventory_path(result, warehouse, n_periods):
    return [result.inventory[(warehouse, t)] for t in range(1, n_periods + 1)]


def test_matches_the_hand_computed_optimum(case, base_case_cost):
    result = _solve(case)

    assert result.is_optimal
    assert result.total_cost == pytest.approx(base_case_cost, abs=1e-6)


def test_holding_costs_discourage_carrying_stock(case):
    # Nothing forces inventory here and holding is not free, so the optimum
    # ships to demand and carries nothing.
    result = _solve(case)

    assert _inventory_path(result, "W1", 3) == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    assert result.holding_cost == pytest.approx(0.0, abs=1e-6)


def test_inventory_recursion_holds_period_by_period(case_factory):
    # Demand rises faster than the plant can produce in the later periods, so
    # the solution has to carry stock and the recursion is worth checking.
    case = case_factory(
        demand={("C1", 1): 10.0, ("C1", 2): 60.0, ("C1", 3): 60.0},
        plant_capacity={"P1": 50.0},
    )
    result = _solve(case)

    opening = case["initial_inventory"]["W1"]
    for t in (1, 2, 3):
        inflow = sum(
            qty for (_, w, tt), qty in result.flows_plant_to_warehouse.items()
            if w == "W1" and tt == t
        )
        outflow = sum(
            qty for (w, _, tt), qty in result.flows_warehouse_to_customer.items()
            if w == "W1" and tt == t
        )
        assert result.inventory[("W1", t)] == pytest.approx(
            opening + inflow - outflow, abs=1e-6
        )
        opening = result.inventory[("W1", t)]


def test_stock_is_pre_built_against_a_production_limit(case_factory):
    # 20 / 150 / 20 against a plant capped at 100 per period. The spike cannot
    # be produced in the period it is needed, so the only way to serve it is to
    # build ahead — which is the whole reason this model spans periods.
    case = case_factory(
        demand={("C1", 1): 20.0, ("C1", 2): 150.0, ("C1", 3): 20.0},
        plant_capacity={"P1": 100.0},
        throughput_capacity={"W1": 200.0},
        storage_capacity={"W1": 100.0},
    )
    result = _solve(case)

    assert result.is_optimal
    # Period 1 produces more than it ships out, banking the difference.
    assert result.inventory[("W1", 1)] == pytest.approx(50.0, abs=1e-6)
    assert result.inventory[("W1", 2)] == pytest.approx(0.0, abs=1e-6)


def test_the_same_spike_is_infeasible_without_storage(case_factory):
    case = case_factory(
        demand={("C1", 1): 20.0, ("C1", 2): 150.0, ("C1", 3): 20.0},
        plant_capacity={"P1": 100.0},
        throughput_capacity={"W1": 200.0},
        storage_capacity={"W1": 0.0},
    )

    with pytest.raises(RuntimeError, match="infeasible"):
        _solve(case)


def test_safety_stock_is_held_in_every_period(case_factory):
    case = case_factory(safety_stock={"W1": 25.0})
    result = _solve(case)

    assert _inventory_path(result, "W1", 3) == pytest.approx([25.0] * 3, abs=1e-6)


def test_safety_stock_is_never_consumed(case_factory):
    # The buffer is built once and then sits there: total production rises by
    # exactly the safety stock, and holding is charged on it every period. It is
    # a standing policy floor, not a reserve the plan is allowed to dip into.
    without = _solve(case_factory())
    with_buffer = _solve(case_factory(safety_stock={"W1": 25.0}))

    produced_without = sum(without.flows_plant_to_warehouse.values())
    produced_with = sum(with_buffer.flows_plant_to_warehouse.values())

    assert produced_without == pytest.approx(30.0, abs=1e-6)  # total demand
    assert produced_with == pytest.approx(55.0, abs=1e-6)  # demand + 25 held back
    # Charged in all three periods, not once.
    assert with_buffer.holding_cost == pytest.approx(0.5 * 25.0 * 3, abs=1e-6)


def test_safety_stock_above_storage_capacity_is_infeasible(case_factory):
    case = case_factory(safety_stock={"W1": 50.0}, storage_capacity={"W1": 30.0})

    with pytest.raises(RuntimeError, match="infeasible"):
        _solve(case)


def test_initial_inventory_is_drawn_down_before_producing(case_factory):
    # Opening stock is free at the start of the horizon, so the plan should use
    # it rather than produce alongside it.
    case = case_factory(initial_inventory={"W1": 10.0})
    result = _solve(case)

    period_one_inflow = sum(
        qty for (_, w, t), qty in result.flows_plant_to_warehouse.items()
        if w == "W1" and t == 1
    )
    assert period_one_inflow == pytest.approx(0.0, abs=1e-6)
    assert sum(result.flows_plant_to_warehouse.values()) == pytest.approx(20.0, abs=1e-6)


def test_demand_is_met_exactly_in_every_period(case_factory):
    case = case_factory(demand={("C1", 1): 7.0, ("C1", 2): 13.0, ("C1", 3): 21.0})
    result = _solve(case)

    for t, expected in ((1, 7.0), (2, 13.0), (3, 21.0)):
        assert result.shipped_to("C1", t) == pytest.approx(expected, abs=1e-6)
