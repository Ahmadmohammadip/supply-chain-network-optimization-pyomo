"""
Typed, validated data structures for the supply chain network model.

Design intent, same as the sibling repos: an instance fails loudly at
construction — a demand series of the wrong length, a safety stock a warehouse
could never hold, aggregate demand beyond what every plant together could make
— rather than surfacing later as an opaque solver infeasibility.

What is deliberately *not* checked here is network-level routability: whether
flow can actually reach every customer given which facilities are candidates.
That is far more expensive to establish than the aggregate checks, and
PROJECT_BRIEF.md section 1.6 explicitly leaves it to the solver. The error
raised by `solve_scn` on infeasibility names it as a likely cause.
"""

from dataclasses import dataclass, field

# Aggregate comparisons are on floats that have usually been through arithmetic,
# so a hair of tolerance keeps an exactly-balanced instance from being rejected.
_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Plant:
    """A candidate production site.

    `fixed_cost` is charged once if the plant opens, for the whole horizon, not
    per period.
    """

    name: str
    capacity: float
    production_cost: float = 0.0
    fixed_cost: float = 0.0

    def __post_init__(self):
        if self.capacity <= 0:
            raise ValueError(f"plant {self.name!r}: capacity must be > 0, got {self.capacity}")
        if self.production_cost < 0:
            raise ValueError(
                f"plant {self.name!r}: production_cost must be >= 0, "
                f"got {self.production_cost}"
            )
        if self.fixed_cost < 0:
            raise ValueError(
                f"plant {self.name!r}: fixed_cost must be >= 0, got {self.fixed_cost}"
            )


@dataclass(frozen=True)
class Warehouse:
    """A candidate distribution site.

    Throughput and storage are separate capacities: `throughput_capacity` caps
    what moves through per period, in each direction independently, while
    `storage_capacity` caps what can be held between periods. Leaving
    `storage_capacity` as None makes it equal to throughput, which recovers the
    single-parameter behavior of PROJECT_BRIEF.md section 1.5. Setting it to
    zero gives a pure cross-dock.

    `safety_stock` is a flat policy input, not a service-level calculation —
    this model has no demand variability to derive one from. Stock held to meet
    it is never drawn down, so it is produced once and charged holding cost in
    every period.
    """

    name: str
    throughput_capacity: float
    storage_capacity: float | None = None
    fixed_cost: float = 0.0
    holding_cost: float = 0.0
    safety_stock: float = 0.0
    initial_inventory: float = 0.0

    def __post_init__(self):
        if self.throughput_capacity <= 0:
            raise ValueError(
                f"warehouse {self.name!r}: throughput_capacity must be > 0, "
                f"got {self.throughput_capacity}"
            )
        if self.storage_capacity is None:
            object.__setattr__(self, "storage_capacity", self.throughput_capacity)
        if self.storage_capacity < 0:
            raise ValueError(
                f"warehouse {self.name!r}: storage_capacity must be >= 0, "
                f"got {self.storage_capacity}"
            )
        for label in ("fixed_cost", "holding_cost", "safety_stock", "initial_inventory"):
            value = getattr(self, label)
            if value < 0:
                raise ValueError(f"warehouse {self.name!r}: {label} must be >= 0, got {value}")

        if self.safety_stock > self.storage_capacity:
            raise ValueError(
                f"warehouse {self.name!r}: safety_stock {self.safety_stock} exceeds "
                f"storage_capacity {self.storage_capacity} — the warehouse could never "
                f"open, since it cannot hold the stock its own policy requires"
            )
        if self.initial_inventory > self.storage_capacity:
            raise ValueError(
                f"warehouse {self.name!r}: initial_inventory {self.initial_inventory} "
                f"exceeds storage_capacity {self.storage_capacity}"
            )

    @property
    def is_cross_dock(self) -> bool:
        """True when the site moves goods but cannot hold them between periods."""
        return self.storage_capacity == 0


@dataclass(frozen=True)
class Customer:
    """A demand point. `demand` holds one figure per period, in order."""

    name: str
    demand: list[float]

    def __post_init__(self):
        if not self.demand:
            raise ValueError(f"customer {self.name!r}: demand must cover at least one period")
        if any(d < 0 for d in self.demand):
            raise ValueError(f"customer {self.name!r}: demand values must be >= 0")
        object.__setattr__(self, "demand", list(self.demand))

    @property
    def total_demand(self) -> float:
        return sum(self.demand)


@dataclass(frozen=True)
class System:
    """A complete instance: candidate facilities, customers, and shipping costs.

    Shipping costs are keyed by name pairs — `(plant, warehouse)` and
    `(warehouse, customer)` — and every pair needs an entry. An arc that should
    not be used is expressed as an expensive one, not a missing one, so that a
    silent typo in a name cannot quietly delete a route.
    """

    plants: list[Plant]
    warehouses: list[Warehouse]
    customers: list[Customer]
    cost_plant_to_warehouse: dict = field(default_factory=dict)
    cost_warehouse_to_customer: dict = field(default_factory=dict)

    def __post_init__(self):
        for label, collection in (
            ("plants", self.plants),
            ("warehouses", self.warehouses),
            ("customers", self.customers),
        ):
            if not collection:
                raise ValueError(f"System must contain at least one entry in {label}")

        names = [f.name for f in (*self.plants, *self.warehouses, *self.customers)]
        if len(names) != len(set(names)):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(
                f"facility and customer names must be unique across the whole "
                f"instance, repeated: {duplicates}"
            )

        lengths = {len(c.demand) for c in self.customers}
        if len(lengths) != 1:
            raise ValueError(
                f"all customers must have the same number of demand periods, "
                f"got lengths: {sorted(lengths)}"
            )

        missing_inbound = [
            (p.name, w.name)
            for p in self.plants
            for w in self.warehouses
            if (p.name, w.name) not in self.cost_plant_to_warehouse
        ]
        if missing_inbound:
            raise ValueError(
                f"cost_plant_to_warehouse is missing {len(missing_inbound)} pair(s), "
                f"e.g. {missing_inbound[:3]}"
            )
        missing_outbound = [
            (w.name, c.name)
            for w in self.warehouses
            for c in self.customers
            if (w.name, c.name) not in self.cost_warehouse_to_customer
        ]
        if missing_outbound:
            raise ValueError(
                f"cost_warehouse_to_customer is missing {len(missing_outbound)} pair(s), "
                f"e.g. {missing_outbound[:3]}"
            )
        if any(v < 0 for v in self.cost_plant_to_warehouse.values()) or any(
            v < 0 for v in self.cost_warehouse_to_customer.values()
        ):
            raise ValueError("shipping costs must be >= 0")

        self._check_aggregate_feasibility()

    def _check_aggregate_feasibility(self) -> None:
        """The cheap necessary conditions from PROJECT_BRIEF.md section 1.6.

        Section 1.6 states the production check per period — demand in any one
        period above total plant capacity. That is **too strict for a model that
        holds inventory** (see PROJECT_BRIEF.md section 8.2): production can be
        banked, so a peak larger than a single period's output is servable by
        building ahead. Applying the per-period rule would reject the `spike`
        sample network, which solves perfectly well.

        What actually holds is the cumulative version: output can be moved
        forward in time but never borrowed from the future, so demand through
        period t must fit within opening stock plus t periods of production.

        Throughput is the opposite case. Every unit consumed in a period must
        leave a warehouse in that same period, so *that* limit really is
        per-period and no amount of pre-building relaxes it.

        Both are necessary, not sufficient: neither can tell whether flow can
        actually route through the network, which is left to the solver.
        """
        opening_stock = sum(w.initial_inventory for w in self.warehouses)
        production_per_period = self.total_plant_capacity
        outbound_per_period = sum(w.throughput_capacity for w in self.warehouses)

        cumulative_demand = 0.0
        for t, period_demand in enumerate(self.demand_by_period, start=1):
            cumulative_demand += period_demand
            available = opening_stock + production_per_period * t

            if cumulative_demand > available + _TOLERANCE:
                raise ValueError(
                    f"demand through period {t} totals {cumulative_demand}, above the "
                    f"{available} available by then ({opening_stock} opening stock plus "
                    f"{t} period(s) at {production_per_period} with every plant open) — "
                    f"no plan can serve it, since this model has no backorders"
                )

            if period_demand > outbound_per_period + _TOLERANCE:
                raise ValueError(
                    f"demand in period {t} is {period_demand}, above the "
                    f"{outbound_per_period} every warehouse together could ship in a "
                    f"period — holding stock in advance cannot help, because the units "
                    f"still have to leave a warehouse in the period they are consumed"
                )

    @property
    def n_periods(self) -> int:
        return len(self.customers[0].demand)

    @property
    def periods(self) -> list[int]:
        return list(range(1, self.n_periods + 1))

    @property
    def total_demand(self) -> float:
        return sum(c.total_demand for c in self.customers)

    @property
    def demand_by_period(self) -> list[float]:
        return [sum(c.demand[t] for c in self.customers) for t in range(self.n_periods)]

    @property
    def total_plant_capacity(self) -> float:
        """Units every candidate plant together could make in one period."""
        return sum(p.capacity for p in self.plants)

    @property
    def total_storage_capacity(self) -> float:
        return sum(w.storage_capacity for w in self.warehouses)

    def model_inputs(self) -> dict:
        """Flatten into the keyword arguments `build_scn_model` takes."""
        return dict(
            plants=[p.name for p in self.plants],
            warehouses=[w.name for w in self.warehouses],
            customers=[c.name for c in self.customers],
            n_periods=self.n_periods,
            demand={
                (c.name, t): c.demand[t - 1] for c in self.customers for t in self.periods
            },
            plant_capacity={p.name: p.capacity for p in self.plants},
            production_cost={p.name: p.production_cost for p in self.plants},
            throughput_capacity={w.name: w.throughput_capacity for w in self.warehouses},
            storage_capacity={w.name: w.storage_capacity for w in self.warehouses},
            holding_cost={w.name: w.holding_cost for w in self.warehouses},
            safety_stock={w.name: w.safety_stock for w in self.warehouses},
            initial_inventory={w.name: w.initial_inventory for w in self.warehouses},
            cost_plant_to_warehouse=dict(self.cost_plant_to_warehouse),
            cost_warehouse_to_customer=dict(self.cost_warehouse_to_customer),
            fixed_cost_plant={p.name: p.fixed_cost for p in self.plants},
            fixed_cost_warehouse={w.name: w.fixed_cost for w in self.warehouses},
        )
