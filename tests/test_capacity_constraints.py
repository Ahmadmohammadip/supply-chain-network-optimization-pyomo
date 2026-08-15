"""Phase 1: plant capacity, and the two warehouse limits the brief folded into
one parameter — throughput and storage."""

import pytest

from scn_opt.model.builder import build_scn_model
from scn_opt.solve import solve_scn


def _solve(case):
    return solve_scn(build_scn_model(**case))


def _inflow(result, warehouse, period):
    return sum(
        qty for (_, w, t), qty in result.flows_plant_to_warehouse.items()
        if w == warehouse and t == period
    )


def _outflow(result, warehouse, period):
    return sum(
        qty for (w, _, t), qty in result.flows_warehouse_to_customer.items()
        if w == warehouse and t == period
    )


def test_plant_capacity_caps_production_per_period(case_factory):
    case = case_factory(
        demand={("C1", 1): 40.0, ("C1", 2): 40.0, ("C1", 3): 40.0},
        plant_capacity={"P1": 45.0},
    )
    result = _solve(case)

    for t in (1, 2, 3):
        assert _inflow(result, "W1", t) <= 45.0 + 1e-6


def test_demand_beyond_total_plant_capacity_is_infeasible(case_factory):
    case = case_factory(
        demand={("C1", 1): 10.0, ("C1", 2): 200.0, ("C1", 3): 10.0},
        plant_capacity={"P1": 50.0},
        storage_capacity={"W1": 20.0},
    )

    with pytest.raises(RuntimeError, match="infeasible"):
        _solve(case)


def test_inbound_throughput_caps_what_a_warehouse_receives(case_factory):
    case = case_factory(
        demand={("C1", 1): 30.0, ("C1", 2): 30.0, ("C1", 3): 30.0},
        throughput_capacity={"W1": 35.0},
    )
    result = _solve(case)

    for t in (1, 2, 3):
        assert _inflow(result, "W1", t) <= 35.0 + 1e-6


def test_outbound_throughput_caps_what_a_warehouse_ships(case_factory):
    # Demand of 40 in one period against an outbound limit of 30 cannot be
    # served however much stock is sitting in the warehouse — pre-building does
    # not help when the constraint is the loading dock, not the shelf.
    case = case_factory(
        demand={("C1", 1): 10.0, ("C1", 2): 40.0, ("C1", 3): 10.0},
        throughput_capacity={"W1": 30.0},
        storage_capacity={"W1": 500.0},
    )

    with pytest.raises(RuntimeError, match="infeasible"):
        _solve(case)


def test_storage_binds_independently_of_throughput(case_factory):
    # The payoff from splitting the brief's single capacity parameter: this
    # warehouse can move 200 a period but hold only 10, so a spike that needs
    # 50 banked in advance fails on storage while throughput is nowhere near
    # its limit.
    spike = dict(
        demand={("C1", 1): 20.0, ("C1", 2): 150.0, ("C1", 3): 20.0},
        plant_capacity={"P1": 100.0},
        throughput_capacity={"W1": 200.0},
    )

    with pytest.raises(RuntimeError, match="infeasible"):
        _solve(case_factory(**spike, storage_capacity={"W1": 10.0}))

    result = _solve(case_factory(**spike, storage_capacity={"W1": 100.0}))
    assert result.is_optimal


def test_a_cross_dock_holds_nothing_but_still_ships(case_factory):
    # storage_capacity=0 with high throughput is a pure cross-dock. It is only
    # expressible because throughput and storage are separate parameters.
    case = case_factory(
        demand={("C1", 1): 50.0, ("C1", 2): 50.0, ("C1", 3): 50.0},
        throughput_capacity={"W1": 100.0},
        storage_capacity={"W1": 0.0},
    )
    result = _solve(case)

    assert result.is_optimal
    for t in (1, 2, 3):
        assert result.inventory[("W1", t)] == pytest.approx(0.0, abs=1e-6)
        assert _inflow(result, "W1", t) == pytest.approx(50.0, abs=1e-6)
        assert _outflow(result, "W1", t) == pytest.approx(50.0, abs=1e-6)


def test_cheaper_warehouse_wins_when_both_have_room(case_factory):
    # Two routes to the same customer, identical except outbound cost. With no
    # capacity pressure the cheap one should take everything.
    case = case_factory(
        warehouses=["W_near", "W_far"],
        throughput_capacity={"W_near": 100.0, "W_far": 100.0},
        storage_capacity={"W_near": 100.0, "W_far": 100.0},
        holding_cost={"W_near": 0.5, "W_far": 0.5},
        safety_stock={"W_near": 0.0, "W_far": 0.0},
        initial_inventory={"W_near": 0.0, "W_far": 0.0},
        cost_plant_to_warehouse={("P1", "W_near"): 2.0, ("P1", "W_far"): 2.0},
        cost_warehouse_to_customer={("W_near", "C1"): 3.0, ("W_far", "C1"): 9.0},
    )
    result = _solve(case)

    assert _outflow(result, "W_near", 1) == pytest.approx(10.0, abs=1e-6)
    assert _outflow(result, "W_far", 1) == pytest.approx(0.0, abs=1e-6)


def test_capacity_pressure_forces_use_of_the_dearer_warehouse(case_factory):
    # Same network, but the cheap warehouse cannot move it all, so the model has
    # to spill the remainder to the expensive one. Asserted on totals rather
    # than a particular split, since ties are common in flow models.
    case = case_factory(
        demand={("C1", 1): 80.0, ("C1", 2): 80.0, ("C1", 3): 80.0},
        warehouses=["W_near", "W_far"],
        throughput_capacity={"W_near": 50.0, "W_far": 100.0},
        storage_capacity={"W_near": 0.0, "W_far": 0.0},
        holding_cost={"W_near": 0.5, "W_far": 0.5},
        safety_stock={"W_near": 0.0, "W_far": 0.0},
        initial_inventory={"W_near": 0.0, "W_far": 0.0},
        cost_plant_to_warehouse={("P1", "W_near"): 2.0, ("P1", "W_far"): 2.0},
        cost_warehouse_to_customer={("W_near", "C1"): 3.0, ("W_far", "C1"): 9.0},
    )
    result = _solve(case)

    for t in (1, 2, 3):
        assert _outflow(result, "W_near", t) == pytest.approx(50.0, abs=1e-6)
        assert _outflow(result, "W_far", t) == pytest.approx(30.0, abs=1e-6)


def test_builder_rejects_a_missing_shipping_cost(case_factory):
    case = case_factory(
        warehouses=["W1", "W2"],
        throughput_capacity={"W1": 100.0, "W2": 100.0},
        storage_capacity={"W1": 100.0, "W2": 100.0},
        holding_cost={"W1": 0.5, "W2": 0.5},
        safety_stock={"W1": 0.0, "W2": 0.0},
        initial_inventory={"W1": 0.0, "W2": 0.0},
        cost_plant_to_warehouse={("P1", "W1"): 2.0},  # W2 missing
        cost_warehouse_to_customer={("W1", "C1"): 3.0, ("W2", "C1"): 3.0},
    )

    with pytest.raises(ValueError, match="cost_plant_to_warehouse is missing"):
        build_scn_model(**case)


def test_builder_rejects_a_missing_demand_entry(case_factory):
    case = case_factory(demand={("C1", 1): 10.0, ("C1", 2): 10.0})  # period 3 missing

    with pytest.raises(ValueError, match="demand is missing"):
        build_scn_model(**case)


def test_builder_rejects_an_empty_echelon(case_factory):
    with pytest.raises(ValueError, match="at least one entry in warehouses"):
        build_scn_model(**case_factory(warehouses=[]))
