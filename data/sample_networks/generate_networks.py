"""Regenerates the sample networks in this directory.

THESE NETWORKS ARE SYNTHETIC. They are hand-designed rather than drawn from any
real supply chain, and small enough that the optimal configuration can be
reasoned about instead of taken on trust. Costs are round numbers chosen to make
a particular trade-off visible, not calibrated to any industry.

Three instances, each making a different point:

* `baseline`   — 2 plants, 3 warehouses, 5 customers, 4 periods (the shape the
                 brief suggests). A regional network where geography decides
                 which warehouses earn their fixed cost.
* `tradeoff`   — the fixed-versus-variable crossover from PROJECT_BRIEF.md
                 section 4, stripped to just that decision so the arithmetic is
                 checkable by hand.
* `spike`      — a demand peak larger than any single period of production, so
                 the plan must build stock ahead. This is the instance that
                 shows why the model spans periods.

The generator is committed alongside the JSON so the data is inspectable rather
than magic, and it builds each network through the schema, so anything it emits
has already passed validation.

Run with:  python data/sample_networks/generate_networks.py
"""

import json
from pathlib import Path

from scn_opt.data.loaders import system_to_dict
from scn_opt.data.schema import Customer, Plant, System, Warehouse

HERE = Path(__file__).parent


def baseline_network() -> System:
    """A regional network: one central warehouse against two local ones.

    The central site can reach everyone but is dearer to open and to ship from;
    the two regional sites are cheap to their own customers and expensive to the
    far side. Which combination wins depends on how demand is spread, which is
    the point of solving it rather than eyeballing it.
    """
    plants = [
        Plant(name="P_north", capacity=120.0, production_cost=1.0, fixed_cost=800.0),
        Plant(name="P_south", capacity=120.0, production_cost=1.2, fixed_cost=600.0),
    ]
    warehouses = [
        Warehouse(
            name="W_central",
            throughput_capacity=150.0,
            storage_capacity=60.0,
            fixed_cost=400.0,
            holding_cost=0.4,
            safety_stock=5.0,
        ),
        Warehouse(
            name="W_east",
            throughput_capacity=80.0,
            storage_capacity=30.0,
            fixed_cost=250.0,
            holding_cost=0.3,
            safety_stock=5.0,
        ),
        Warehouse(
            name="W_west",
            throughput_capacity=80.0,
            storage_capacity=30.0,
            fixed_cost=250.0,
            holding_cost=0.3,
            safety_stock=5.0,
        ),
    ]
    customers = [
        Customer(name="C_east_1", demand=[30.0, 35.0, 25.0, 30.0]),
        Customer(name="C_east_2", demand=[20.0, 25.0, 20.0, 15.0]),
        Customer(name="C_west_1", demand=[25.0, 20.0, 30.0, 25.0]),
        Customer(name="C_west_2", demand=[15.0, 20.0, 15.0, 20.0]),
        Customer(name="C_middle", demand=[10.0, 15.0, 20.0, 10.0]),
    ]

    inbound = {
        ("P_north", "W_central"): 2.0,
        ("P_north", "W_east"): 1.5,
        ("P_north", "W_west"): 3.5,
        ("P_south", "W_central"): 2.0,
        ("P_south", "W_east"): 3.5,
        ("P_south", "W_west"): 1.5,
    }
    outbound = {
        # The central site reaches everyone at a middling price.
        ("W_central", "C_east_1"): 4.0,
        ("W_central", "C_east_2"): 4.0,
        ("W_central", "C_west_1"): 4.0,
        ("W_central", "C_west_2"): 4.0,
        ("W_central", "C_middle"): 1.0,
        # The regional sites are cheap nearby and dear across the map.
        ("W_east", "C_east_1"): 1.0,
        ("W_east", "C_east_2"): 1.5,
        ("W_east", "C_west_1"): 8.0,
        ("W_east", "C_west_2"): 8.5,
        ("W_east", "C_middle"): 4.5,
        ("W_west", "C_east_1"): 8.5,
        ("W_west", "C_east_2"): 8.0,
        ("W_west", "C_west_1"): 1.0,
        ("W_west", "C_west_2"): 1.5,
        ("W_west", "C_middle"): 4.5,
    }
    return System(
        plants=plants,
        warehouses=warehouses,
        customers=customers,
        cost_plant_to_warehouse=inbound,
        cost_warehouse_to_customer=outbound,
    )


def tradeoff_network(demand_per_period: float = 57.0, n_periods: int = 4) -> System:
    """PROJECT_BRIEF.md section 4, reduced to the decision it is about.

    Everything except the two levers costs nothing, so the comparison is:

        W_cheap_fixed:     100 + 5 x total units
        W_cheap_shipping: 1000 + 1 x total units

    Equal at 100 + 20D = 1000 + 4D over four periods, i.e. D = 56.25. The
    committed instance sits just above at 57, so the cheap-to-ship site wins;
    drop it to 56 and the decision flips.
    """
    return System(
        plants=[Plant(name="P1", capacity=10_000.0)],
        warehouses=[
            Warehouse(name="W_cheap_fixed", throughput_capacity=10_000.0, fixed_cost=100.0),
            Warehouse(
                name="W_cheap_shipping", throughput_capacity=10_000.0, fixed_cost=1000.0
            ),
        ],
        customers=[Customer(name="C1", demand=[demand_per_period] * n_periods)],
        cost_plant_to_warehouse={
            ("P1", "W_cheap_fixed"): 0.0,
            ("P1", "W_cheap_shipping"): 0.0,
        },
        cost_warehouse_to_customer={
            ("W_cheap_fixed", "C1"): 5.0,
            ("W_cheap_shipping", "C1"): 1.0,
        },
    )


def spike_network() -> System:
    """A peak bigger than one period of production.

    Total demand across the horizon is well within what the plant can make, but
    period 2 alone is not — so the only way to serve it is to produce early and
    hold. Storage is the binding resource here, not throughput.
    """
    return System(
        plants=[Plant(name="P1", capacity=100.0, production_cost=1.0, fixed_cost=200.0)],
        warehouses=[
            Warehouse(
                name="W1",
                throughput_capacity=200.0,
                storage_capacity=100.0,
                fixed_cost=150.0,
                holding_cost=0.5,
            )
        ],
        customers=[Customer(name="C1", demand=[20.0, 150.0, 20.0, 20.0])],
        cost_plant_to_warehouse={("P1", "W1"): 1.0},
        cost_warehouse_to_customer={("W1", "C1"): 2.0},
    )


NETWORKS = {
    "baseline": baseline_network,
    "tradeoff": tradeoff_network,
    "spike": spike_network,
}


def main() -> None:
    for name, factory in NETWORKS.items():
        system = factory()
        payload = system_to_dict(system)
        payload = {
            "name": name,
            "description": (factory.__doc__ or "").strip().splitlines()[0],
            **payload,
        }
        path = HERE / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(
            f"wrote {path.name}: {len(system.plants)} plant(s), "
            f"{len(system.warehouses)} warehouse(s), {len(system.customers)} customer(s), "
            f"{system.n_periods} periods, total demand {system.total_demand:g}"
        )


if __name__ == "__main__":
    main()
