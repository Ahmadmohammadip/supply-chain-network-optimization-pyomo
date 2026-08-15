"""End-to-end runs on the committed sample networks.

These check the properties any valid plan must have, and the specific point each
instance was designed to make. They assert on costs, totals, and which
facilities open — not on individual flow splits, since ties between equal-cost
routings are common in network flow models and pinning one is a test that breaks
for no good reason.
"""

from pathlib import Path

import pytest

from scn_opt.data.loaders import load_system_json
from scn_opt.model.builder import build_from_system
from scn_opt.solve import solve_scn

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_networks"
NETWORKS = ["baseline", "tradeoff", "spike"]


def _load(name):
    return load_system_json(SAMPLE_DIR / f"{name}.json")


def _solve(name, **kwargs):
    system = _load(name)
    return system, solve_scn(build_from_system(system, **kwargs))


@pytest.mark.parametrize("name", NETWORKS)
def test_sample_network_loads_and_validates(name):
    system = _load(name)

    assert system.n_periods >= 1
    assert system.plants and system.warehouses and system.customers


@pytest.mark.parametrize("name", NETWORKS)
def test_sample_network_solves_to_a_valid_plan(name):
    system, result = _solve(name)

    assert result.is_optimal
    assert result.open_plants  # something has to make the goods
    assert result.open_warehouses

    # Demand met exactly, every customer, every period.
    for customer in system.customers:
        for t in system.periods:
            assert result.shipped_to(customer.name, t) == pytest.approx(
                customer.demand[t - 1], abs=1e-6
            )


@pytest.mark.parametrize("name", NETWORKS)
def test_cost_breakdown_sums_to_the_objective(name):
    _, result = _solve(name)

    assert sum(result.cost_breakdown().values()) == pytest.approx(
        result.total_cost, abs=1e-6
    )


@pytest.mark.parametrize("name", NETWORKS)
def test_no_flow_or_stock_at_a_closed_facility(name):
    system, result = _solve(name)

    closed_plants = {p.name for p in system.plants} - set(result.open_plants)
    closed_warehouses = {w.name for w in system.warehouses} - set(result.open_warehouses)

    for p in closed_plants:
        assert not any(key[0] == p for key in result.flows_plant_to_warehouse)
    for w in closed_warehouses:
        assert not any(key[1] == w for key in result.flows_plant_to_warehouse)
        assert not any(key[0] == w for key in result.flows_warehouse_to_customer)
        for t in system.periods:
            assert result.inventory[(w, t)] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("name", NETWORKS)
def test_inventory_recursion_holds(name):
    system, result = _solve(name)

    for warehouse in system.warehouses:
        opening = warehouse.initial_inventory
        for t in system.periods:
            inflow = sum(
                qty for (_, w, tt), qty in result.flows_plant_to_warehouse.items()
                if w == warehouse.name and tt == t
            )
            outflow = sum(
                qty for (w, _, tt), qty in result.flows_warehouse_to_customer.items()
                if w == warehouse.name and tt == t
            )
            assert result.inventory[(warehouse.name, t)] == pytest.approx(
                opening + inflow - outflow, abs=1e-6
            )
            opening = result.inventory[(warehouse.name, t)]


@pytest.mark.parametrize("name", NETWORKS)
def test_capacities_are_respected(name):
    system, result = _solve(name)

    for plant in system.plants:
        for t in system.periods:
            produced = sum(
                qty for (p, _, tt), qty in result.flows_plant_to_warehouse.items()
                if p == plant.name and tt == t
            )
            assert produced <= plant.capacity + 1e-6

    for warehouse in system.warehouses:
        for t in system.periods:
            inbound = sum(
                qty for (_, w, tt), qty in result.flows_plant_to_warehouse.items()
                if w == warehouse.name and tt == t
            )
            outbound = sum(
                qty for (w, _, tt), qty in result.flows_warehouse_to_customer.items()
                if w == warehouse.name and tt == t
            )
            assert inbound <= warehouse.throughput_capacity + 1e-6
            assert outbound <= warehouse.throughput_capacity + 1e-6
            assert result.inventory[(warehouse.name, t)] <= warehouse.storage_capacity + 1e-6


@pytest.mark.parametrize("name", NETWORKS)
def test_optimizing_never_costs_more_than_opening_everything(name):
    _, optimized = _solve(name)
    _, everything = _solve(name, force_open_all=True)

    assert everything.total_cost >= optimized.total_cost - 1e-6


def test_baseline_prefers_the_regional_pair_over_the_central_site():
    # The point of the baseline network: two cheap local warehouses beat one
    # central one that can reach everybody, because most demand sits near the
    # regions. Optimizing is worth several hundred against running everything.
    _, optimized = _solve("baseline")
    _, everything = _solve("baseline", force_open_all=True)

    assert set(optimized.open_warehouses) == {"W_east", "W_west"}
    assert everything.total_cost - optimized.total_cost > 100.0


def test_tradeoff_network_sits_just_past_the_crossover():
    # Committed at 57 units a period, just above the hand-computed 56.25, so the
    # expensive-to-open, cheap-to-ship site wins by 8.
    _, result = _solve("tradeoff")

    assert result.open_warehouses == ["W_cheap_shipping"]
    assert result.total_cost == pytest.approx(1000 + 4 * 57, abs=1e-6)


def test_spike_network_builds_stock_ahead_of_the_peak():
    # 150 in period 2 against a plant that makes 100 a period: the only way
    # through is to produce early and hold. This is the instance that shows why
    # the model spans periods rather than solving each in isolation.
    system, result = _solve("spike")

    peak = max(system.demand_by_period)
    assert peak > system.total_plant_capacity  # unservable from that period alone
    assert result.inventory[("W1", 1)] == pytest.approx(50.0, abs=1e-6)
    assert result.holding_cost > 0


def _rescaled(system, demand: float = 1.0, fixed: float = 1.0):
    """The baseline network with demand and/or fixed costs scaled.

    Mirrors what the Streamlit app's sliders do, so the scenarios the app can
    reach are the ones covered here.
    """
    import json

    from scn_opt.data.loaders import system_from_dict, system_to_dict

    data = system_to_dict(system)
    for plant in data["plants"]:
        plant["fixed_cost"] *= fixed
    for warehouse in data["warehouses"]:
        warehouse["fixed_cost"] *= fixed
    for customer in data["customers"]:
        customer["demand"] = [d * demand for d in customer["demand"]]
    return system_from_dict(json.loads(json.dumps(data)))


def test_free_facilities_are_all_worth_opening():
    # With opening free, there is nothing to trade against the shipping saving,
    # so every candidate should be kept and optimizing should save nothing.
    system = _rescaled(_load("baseline"), fixed=0.0)

    optimized = solve_scn(build_from_system(system))
    everything = solve_scn(build_from_system(system, force_open_all=True))

    assert len(optimized.open_plants) == len(system.plants)
    assert len(optimized.open_warehouses) == len(system.warehouses)
    assert optimized.total_cost == pytest.approx(everything.total_cost, abs=1e-6)


def test_dearer_facilities_push_the_network_smaller():
    # Raising the cost of opening should never grow the network.
    cheap = _rescaled(_load("baseline"), fixed=0.5)
    dear = _rescaled(_load("baseline"), fixed=3.0)

    cheap_result = solve_scn(build_from_system(cheap))
    dear_result = solve_scn(build_from_system(dear))

    assert len(dear_result.open_warehouses) <= len(cheap_result.open_warehouses)
    assert len(dear_result.open_plants) <= len(cheap_result.open_plants)


def test_enough_volume_justifies_the_whole_network():
    # The other direction: at double demand every candidate facility earns its
    # keep, so the optimized network and the open-everything counterfactual
    # coincide. Scale is what moves the strategic answer.
    system = _rescaled(_load("baseline"), demand=2.0)

    optimized = solve_scn(build_from_system(system))
    everything = solve_scn(build_from_system(system, force_open_all=True))

    assert len(optimized.open_warehouses) == len(system.warehouses)
    assert optimized.total_cost == pytest.approx(everything.total_cost, abs=1e-6)
