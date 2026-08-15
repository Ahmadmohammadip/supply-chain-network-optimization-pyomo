"""Phase 3: instance validation.

Catching a malformed instance at construction matters because the alternative
is an opaque "infeasible" from the solver with nothing pointing at the cause.
"""

import re

import pytest

from scn_opt.data.schema import Customer, Plant, System, Warehouse


def _system(**overrides) -> System:
    defaults = dict(
        plants=[Plant(name="P1", capacity=100.0, production_cost=1.0, fixed_cost=500.0)],
        warehouses=[Warehouse(name="W1", throughput_capacity=80.0, storage_capacity=40.0)],
        customers=[Customer(name="C1", demand=[10.0, 20.0, 30.0])],
        cost_plant_to_warehouse={("P1", "W1"): 2.0},
        cost_warehouse_to_customer={("W1", "C1"): 3.0},
    )
    defaults.update(overrides)
    return System(**defaults)


def test_valid_instance_reports_its_derived_values():
    system = _system()

    assert system.n_periods == 3
    assert system.periods == [1, 2, 3]
    assert system.total_demand == pytest.approx(60.0)
    assert system.demand_by_period == pytest.approx([10.0, 20.0, 30.0])
    assert system.total_plant_capacity == pytest.approx(100.0)
    assert system.total_storage_capacity == pytest.approx(40.0)


def test_storage_capacity_defaults_to_throughput():
    # Recovers the single-parameter behavior of brief section 1.5.
    warehouse = Warehouse(name="W1", throughput_capacity=60.0)

    assert warehouse.storage_capacity == pytest.approx(60.0)
    assert not warehouse.is_cross_dock


def test_zero_storage_is_a_cross_dock():
    warehouse = Warehouse(name="W1", throughput_capacity=60.0, storage_capacity=0.0)

    assert warehouse.is_cross_dock


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"capacity": 0.0}, "capacity must be > 0"),
        ({"production_cost": -1.0}, "production_cost must be >= 0"),
        ({"fixed_cost": -5.0}, "fixed_cost must be >= 0"),
    ],
)
def test_invalid_plant_is_rejected(kwargs, expected):
    with pytest.raises(ValueError, match=re.escape(expected)):
        Plant(name="P1", **{"capacity": 10.0, **kwargs})


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"throughput_capacity": 0.0}, "throughput_capacity must be > 0"),
        ({"storage_capacity": -1.0}, "storage_capacity must be >= 0"),
        ({"holding_cost": -0.5}, "holding_cost must be >= 0"),
        ({"safety_stock": -1.0}, "safety_stock must be >= 0"),
    ],
)
def test_invalid_warehouse_is_rejected(kwargs, expected):
    with pytest.raises(ValueError, match=re.escape(expected)):
        Warehouse(name="W1", **{"throughput_capacity": 10.0, **kwargs})


def test_safety_stock_above_storage_is_rejected():
    # This warehouse could never open: it cannot hold the stock its own policy
    # demands. Better caught here than as a bare "infeasible" later.
    with pytest.raises(ValueError, match="could never open"):
        Warehouse(name="W1", throughput_capacity=100.0, storage_capacity=20.0, safety_stock=50.0)


def test_initial_inventory_above_storage_is_rejected():
    with pytest.raises(ValueError, match="initial_inventory 60.0 exceeds"):
        Warehouse(
            name="W1",
            throughput_capacity=100.0,
            storage_capacity=50.0,
            initial_inventory=60.0,
        )


def test_empty_demand_is_rejected():
    with pytest.raises(ValueError, match="at least one period"):
        Customer(name="C1", demand=[])


def test_negative_demand_is_rejected():
    with pytest.raises(ValueError, match="demand values must be >= 0"):
        Customer(name="C1", demand=[10.0, -5.0])


@pytest.mark.parametrize("field", ["plants", "warehouses", "customers"])
def test_empty_echelon_is_rejected(field):
    with pytest.raises(ValueError, match=f"at least one entry in {field}"):
        _system(**{field: []})


def test_duplicate_names_across_echelons_are_rejected():
    # Names key the shipping-cost maps, so a collision between a warehouse and
    # a customer would silently corrupt the network rather than error.
    with pytest.raises(ValueError, match=r"names must be unique.*\['W1'\]"):
        _system(
            customers=[Customer(name="W1", demand=[10.0, 20.0, 30.0])],
            cost_warehouse_to_customer={("W1", "W1"): 3.0},
        )


def test_mismatched_demand_lengths_are_rejected():
    with pytest.raises(ValueError, match=r"same number of demand periods.*\[2, 3\]"):
        _system(
            customers=[
                Customer(name="C1", demand=[10.0, 20.0, 30.0]),
                Customer(name="C2", demand=[5.0, 5.0]),
            ],
            cost_warehouse_to_customer={("W1", "C1"): 3.0, ("W1", "C2"): 4.0},
        )


def test_missing_inbound_arc_cost_is_rejected():
    with pytest.raises(ValueError, match="cost_plant_to_warehouse is missing 1 pair"):
        _system(cost_plant_to_warehouse={})


def test_missing_outbound_arc_cost_is_rejected():
    with pytest.raises(ValueError, match="cost_warehouse_to_customer is missing 1 pair"):
        _system(cost_warehouse_to_customer={})


def test_negative_shipping_cost_is_rejected():
    with pytest.raises(ValueError, match="shipping costs must be >= 0"):
        _system(cost_plant_to_warehouse={("P1", "W1"): -2.0})


def test_demand_beyond_total_plant_capacity_is_rejected():
    # Per period, not over the horizon: capacity is a rate, so a peak can be
    # unservable even when the totals look comfortable.
    with pytest.raises(ValueError, match="demand in period 2 is 150.0, above the 100.0"):
        _system(customers=[Customer(name="C1", demand=[10.0, 150.0, 10.0])])


def test_demand_exactly_at_capacity_is_accepted():
    # The tolerance exists so an exactly-balanced instance is not rejected by
    # floating-point noise.
    system = _system(customers=[Customer(name="C1", demand=[100.0, 100.0, 100.0])])

    assert system.demand_by_period == pytest.approx([100.0] * 3)


def test_model_inputs_cover_every_builder_argument():
    from scn_opt.model.builder import build_scn_model

    inputs = _system().model_inputs()

    # Anything the builder requires and the schema forgets would show up here as
    # a TypeError rather than as a confusing failure deeper in.
    build_scn_model(**inputs)

    assert inputs["n_periods"] == 3
    assert inputs["demand"][("C1", 2)] == pytest.approx(20.0)
    assert inputs["storage_capacity"]["W1"] == pytest.approx(40.0)
    assert inputs["fixed_cost_plant"]["P1"] == pytest.approx(500.0)
