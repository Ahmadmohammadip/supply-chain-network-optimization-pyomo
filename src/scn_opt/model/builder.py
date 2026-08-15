"""
Builds a Pyomo ConcreteModel for multi-echelon supply chain network planning.

The complete model: a strategic layer deciding which plants and warehouses to
open, and an operational layer deciding production, shipping, and inventory
across a multi-period horizon — solved together as one MILP rather than in
sequence. docs/formulation.md carries the derivation and citations.

    min   fixed + production + inbound shipping + outbound shipping + holding
    s.t.  sum_w ship_pw[p,w,t] <= plant_capacity[p] * open_plant[p]
          I[w,t] = I[w,t-1] + sum_p ship_pw - sum_c ship_wc
          sum_p ship_pw[p,w,t] <= throughput_capacity[w] * open_warehouse[w]
          sum_c ship_wc[w,c,t] <= throughput_capacity[w] * open_warehouse[w]
          I[w,t] <= storage_capacity[w] * open_warehouse[w]
          I[w,t] >= safety_stock[w] * open_warehouse[w]
          sum_w ship_wc[w,c,t] = demand[c,t]

## Facilities are open for the whole horizon

`open_plant` and `open_warehouse` carry no time index: a facility is open for
the entire horizon or not at all. Phased opening and closing would need
time-indexed binaries and is out of scope (PROJECT_BRIEF.md section 5).

One consequence worth knowing: fixed cost is charged **once**, not per period,
so the horizon length is itself a lever on the strategic/operational balance.
A longer horizon accumulates more variable cost against the same one-off fixed
cost, which makes the model readier to pay for a facility that ships cheaply.

## Capacity does the work of big-M

Every open/close binary multiplies a real capacity, never an arbitrary large
constant. That is deliberate — a loose big-M weakens the linear relaxation and
can cause numerical trouble; the capacity is both the tightest valid bound and
the physically meaningful one.

## Throughput and storage are separate capacities

PROJECT_BRIEF.md section 1.5 uses a single parameter for a warehouse's inbound
limit, outbound limit, and storage capacity all at once. This model splits it
in two (see PROJECT_BRIEF.md section 8.1). A warehouse is rated for how much it
can move per period and, separately, for how much it can keep — which is what
distinguishes a cross-dock (high throughput, no storage) from a holding depot.

## Demand is met exactly

No backorders, no lost sales. An instance whose demand cannot be served comes
back infeasible rather than merely expensive, which is why the schema checks
aggregate capacity against aggregate demand before the solver ever sees it.
"""

from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    Param,
    Set,
    Var,
    minimize,
)


def build_scn_model(
    plants: list,
    warehouses: list,
    customers: list,
    n_periods: int,
    demand: dict,
    plant_capacity: dict,
    production_cost: dict,
    throughput_capacity: dict,
    storage_capacity: dict,
    holding_cost: dict,
    safety_stock: dict,
    initial_inventory: dict,
    cost_plant_to_warehouse: dict,
    cost_warehouse_to_customer: dict,
    fixed_cost_plant: dict | None = None,
    fixed_cost_warehouse: dict | None = None,
    force_open_all: bool = False,
) -> ConcreteModel:
    """Build the network model from primitive inputs.

    `demand` is keyed by (customer, period) with periods 1..n_periods; the two
    cost maps are keyed by (plant, warehouse) and (warehouse, customer) and must
    cover every pair. Fixed costs default to zero, which makes opening free and
    reduces the model to its operational layer.

    `force_open_all` pins every facility open while still charging its fixed
    cost, giving the counterfactual "what if we ran the whole candidate
    network?" against which the optimized network can be compared. Because it
    fixes the binaries rather than removing them, the two objectives are
    directly comparable.

    Callers holding a validated `System` should use `build_from_system`, which
    unpacks it and calls this.
    """
    if n_periods < 1:
        raise ValueError(f"n_periods must be >= 1, got {n_periods}")
    for label, collection in (
        ("plants", plants),
        ("warehouses", warehouses),
        ("customers", customers),
    ):
        if not collection:
            raise ValueError(f"model needs at least one entry in {label}")

    fixed_cost_plant = fixed_cost_plant or dict.fromkeys(plants, 0.0)
    fixed_cost_warehouse = fixed_cost_warehouse or dict.fromkeys(warehouses, 0.0)

    periods = list(range(1, n_periods + 1))
    _require_keys(demand, [(c, t) for c in customers for t in periods], "demand")
    _require_keys(
        cost_plant_to_warehouse,
        [(p, w) for p in plants for w in warehouses],
        "cost_plant_to_warehouse",
    )
    _require_keys(
        cost_warehouse_to_customer,
        [(w, c) for w in warehouses for c in customers],
        "cost_warehouse_to_customer",
    )
    _require_keys(fixed_cost_plant, list(plants), "fixed_cost_plant")
    _require_keys(fixed_cost_warehouse, list(warehouses), "fixed_cost_warehouse")

    m = ConcreteModel(name="SupplyChainNetwork")

    # --- Sets ---
    m.P = Set(initialize=plants, ordered=True)
    m.W = Set(initialize=warehouses, ordered=True)
    m.C = Set(initialize=customers, ordered=True)
    m.T = Set(initialize=periods, ordered=True)

    # --- Parameters ---
    m.demand = Param(m.C, m.T, initialize=dict(demand))
    m.plant_capacity = Param(m.P, initialize=dict(plant_capacity))
    m.production_cost = Param(m.P, initialize=dict(production_cost))
    m.throughput_capacity = Param(m.W, initialize=dict(throughput_capacity))
    m.storage_capacity = Param(m.W, initialize=dict(storage_capacity))
    m.holding_cost = Param(m.W, initialize=dict(holding_cost))
    m.safety_stock = Param(m.W, initialize=dict(safety_stock))
    m.initial_inventory = Param(m.W, initialize=dict(initial_inventory))
    m.cost_pw = Param(m.P, m.W, initialize=dict(cost_plant_to_warehouse))
    m.cost_wc = Param(m.W, m.C, initialize=dict(cost_warehouse_to_customer))
    m.fixed_cost_plant = Param(m.P, initialize=dict(fixed_cost_plant))
    m.fixed_cost_warehouse = Param(m.W, initialize=dict(fixed_cost_warehouse))

    # --- Variables ---
    m.open_plant = Var(m.P, domain=Binary)
    m.open_warehouse = Var(m.W, domain=Binary)
    m.ship_pw = Var(m.P, m.W, m.T, domain=NonNegativeReals)
    m.ship_wc = Var(m.W, m.C, m.T, domain=NonNegativeReals)
    m.inventory = Var(m.W, m.T, domain=NonNegativeReals)

    if force_open_all:
        for p in plants:
            m.open_plant[p].fix(1)
        for w in warehouses:
            m.open_warehouse[w].fix(1)

    # --- Objective ---
    # Held as named expressions so solve.py can report a cost breakdown that
    # provably sums to the objective, rather than recomputing it independently
    # and hoping the two agree.
    m.fixed_expr = sum(
        m.fixed_cost_plant[p] * m.open_plant[p] for p in m.P
    ) + sum(m.fixed_cost_warehouse[w] * m.open_warehouse[w] for w in m.W)
    m.production_expr = sum(
        m.production_cost[p] * m.ship_pw[p, w, t] for p in m.P for w in m.W for t in m.T
    )
    m.inbound_expr = sum(
        m.cost_pw[p, w] * m.ship_pw[p, w, t] for p in m.P for w in m.W for t in m.T
    )
    m.outbound_expr = sum(
        m.cost_wc[w, c] * m.ship_wc[w, c, t] for w in m.W for c in m.C for t in m.T
    )
    m.holding_expr = sum(
        m.holding_cost[w] * m.inventory[w, t] for w in m.W for t in m.T
    )

    m.total_cost = Objective(
        expr=m.fixed_expr
        + m.production_expr
        + m.inbound_expr
        + m.outbound_expr
        + m.holding_expr,
        sense=minimize,
    )

    # --- Constraints ---

    def _plant_capacity_rule(m, p, t):
        return sum(m.ship_pw[p, w, t] for w in m.W) <= m.plant_capacity[p] * m.open_plant[p]

    m.plant_capacity_con = Constraint(m.P, m.T, rule=_plant_capacity_rule)

    # Everything shipped in arrives the same period it leaves the plant: transit
    # lead times are out of scope.
    def _inventory_balance_rule(m, w, t):
        opening = (
            m.initial_inventory[w] if t == m.T.first() else m.inventory[w, m.T.prev(t)]
        )
        inflow = sum(m.ship_pw[p, w, t] for p in m.P)
        outflow = sum(m.ship_wc[w, c, t] for c in m.C)
        return m.inventory[w, t] == opening + inflow - outflow

    m.inventory_balance_con = Constraint(m.W, m.T, rule=_inventory_balance_rule)

    def _inbound_throughput_rule(m, w, t):
        return (
            sum(m.ship_pw[p, w, t] for p in m.P)
            <= m.throughput_capacity[w] * m.open_warehouse[w]
        )

    m.inbound_throughput_con = Constraint(m.W, m.T, rule=_inbound_throughput_rule)

    def _outbound_throughput_rule(m, w, t):
        return (
            sum(m.ship_wc[w, c, t] for c in m.C)
            <= m.throughput_capacity[w] * m.open_warehouse[w]
        )

    m.outbound_throughput_con = Constraint(m.W, m.T, rule=_outbound_throughput_rule)

    def _storage_rule(m, w, t):
        return m.inventory[w, t] <= m.storage_capacity[w] * m.open_warehouse[w]

    m.storage_con = Constraint(m.W, m.T, rule=_storage_rule)

    # A flat policy floor, not a service-level calculation — this model has no
    # demand variability to derive one from. Note the consequence: stock held to
    # satisfy this is never drawn down, so it is produced once and then charged
    # holding cost in every period.
    #
    # Gated on the binary so it only applies to warehouses that actually open;
    # a closed one is already forced to zero inventory by the storage bound.
    def _safety_stock_rule(m, w, t):
        return m.inventory[w, t] >= m.safety_stock[w] * m.open_warehouse[w]

    m.safety_stock_con = Constraint(m.W, m.T, rule=_safety_stock_rule)

    def _demand_rule(m, c, t):
        return sum(m.ship_wc[w, c, t] for w in m.W) == m.demand[c, t]

    m.demand_con = Constraint(m.C, m.T, rule=_demand_rule)

    return m


def _require_keys(mapping: dict, required: list, label: str) -> None:
    missing = [k for k in required if k not in mapping]
    if missing:
        raise ValueError(
            f"{label} is missing {len(missing)} entries, e.g. {missing[:3]} — "
            f"every combination needs a value"
        )


def build_from_system(system, force_open_all: bool = False) -> ConcreteModel:
    """Build the model from a validated `System` (see data/schema.py).

    This is the entry point most callers want. The primitive `build_scn_model`
    stays public because it predates the schema and is useful for exercising the
    formulation against hand-built instances with no file format involved.
    """
    return build_scn_model(**system.model_inputs(), force_open_all=force_open_all)
