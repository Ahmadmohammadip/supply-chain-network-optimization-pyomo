"""Solver interface and result extraction.

This is a MILP once the facility binaries arrive in phase 2, so "the solver
stopped" and "the solver proved an optimum" are different events and the result
type has to say which happened — the same convention the sibling
`cvrp-optimization-pyomo` repo settled on.
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from pyomo.environ import ConcreteModel, SolverFactory, value
from pyomo.opt import TerminationCondition

DEFAULT_SOLVER = "appsi_highs"

# Flows below this are solver noise rather than real shipments.
FLOW_TOLERANCE = 1e-6


@dataclass
class NetworkResult:
    """A solved (or time-limited) network plan.

    Flow and inventory dictionaries are keyed by the same tuples the model uses:
    (plant, warehouse, period), (warehouse, customer, period), and
    (warehouse, period).
    """

    flows_plant_to_warehouse: dict
    flows_warehouse_to_customer: dict
    inventory: dict
    total_cost: float
    production_cost: float
    inbound_cost: float
    outbound_cost: float
    holding_cost: float
    is_optimal: bool
    gap: float
    solve_time: float
    termination: str
    fixed_cost: float = 0.0
    open_plants: list = field(default_factory=list)
    open_warehouses: list = field(default_factory=list)

    def cost_breakdown(self) -> dict:
        """Cost by component. Sums to `total_cost`."""
        return {
            "fixed": self.fixed_cost,
            "production": self.production_cost,
            "inbound_shipping": self.inbound_cost,
            "outbound_shipping": self.outbound_cost,
            "holding": self.holding_cost,
        }

    def shipped_to(self, customer, period: int) -> float:
        return sum(
            qty
            for (_, c, t), qty in self.flows_warehouse_to_customer.items()
            if c == customer and t == period
        )

    def summary(self) -> str:
        status = (
            "proven optimal"
            if self.is_optimal
            else f"feasible, not proven optimal (gap {self.gap:.2%})"
        )
        network = (
            f"{len(self.open_plants)} plant(s), {len(self.open_warehouses)} warehouse(s)"
        )
        return f"{self.total_cost:,.2f} total cost — {network} — {status}, {self.solve_time:.1f}s"


def solve_scn(
    model: ConcreteModel,
    solver_name: str = DEFAULT_SOLVER,
    time_limit: float | None = None,
) -> NetworkResult:
    """Solve the model and extract flows, inventory, and the cost breakdown.

    Hitting `time_limit` is not an error: if a feasible plan was found it comes
    back with `is_optimal=False` and a gap. Raises when there is genuinely
    nothing to return — an infeasible instance, or a limit reached before any
    feasible plan was found.
    """
    solver = SolverFactory(solver_name)
    if time_limit is not None:
        solver.options["time_limit"] = time_limit

    start = time.perf_counter()
    # load_solutions=False so a suboptimal incumbent is loaded deliberately
    # rather than raising, or warning, on our behalf.
    results = solver.solve(model, load_solutions=False)
    elapsed = time.perf_counter() - start

    condition = results.solver.termination_condition
    lower_bound = _as_float(results.problem.lower_bound)
    upper_bound = _as_float(results.problem.upper_bound)
    has_incumbent = upper_bound is not None and abs(upper_bound) != float("inf")

    if condition == TerminationCondition.infeasible:
        raise RuntimeError(
            "Instance is infeasible. Common causes: demand in some period exceeds "
            "what open plants can produce, a customer no open warehouse can reach, "
            "or a safety stock above a warehouse's storage capacity. A validated "
            "System catches the aggregate cases before the solve."
        )
    if condition not in (TerminationCondition.optimal, TerminationCondition.maxTimeLimit):
        raise RuntimeError(f"Solve failed with {solver_name}: termination = {condition}")
    if not has_incumbent:
        raise RuntimeError(
            f"Solver stopped ({condition}) without finding any feasible plan"
            + (f" within {time_limit}s" if time_limit else "")
            + "."
        )

    with _quiet_pyomo_solution_loading():
        model.solutions.load_from(results)

    flows_pw = {
        (p, w, t): value(model.ship_pw[p, w, t])
        for p in model.P
        for w in model.W
        for t in model.T
        if value(model.ship_pw[p, w, t]) > FLOW_TOLERANCE
    }
    flows_wc = {
        (w, c, t): value(model.ship_wc[w, c, t])
        for w in model.W
        for c in model.C
        for t in model.T
        if value(model.ship_wc[w, c, t]) > FLOW_TOLERANCE
    }
    inventory = {
        (w, t): value(model.inventory[w, t]) for w in model.W for t in model.T
    }

    total_cost = value(model.total_cost)
    is_optimal = condition == TerminationCondition.optimal

    return NetworkResult(
        flows_plant_to_warehouse=flows_pw,
        flows_warehouse_to_customer=flows_wc,
        inventory=inventory,
        total_cost=total_cost,
        production_cost=value(model.production_expr),
        inbound_cost=value(model.inbound_expr),
        outbound_cost=value(model.outbound_expr),
        holding_cost=value(model.holding_expr),
        fixed_cost=value(model.fixed_expr) if hasattr(model, "fixed_expr") else 0.0,
        open_plants=_open_facilities(model, "open_plant", model.P),
        open_warehouses=_open_facilities(model, "open_warehouse", model.W),
        is_optimal=is_optimal,
        gap=_relative_gap(lower_bound, total_cost),
        solve_time=elapsed,
        termination=str(condition),
    )


def _open_facilities(model: ConcreteModel, attribute: str, index_set) -> list:
    """Which facilities are open.

    Before the binaries exist (phase 1) every facility is open by construction,
    so the whole index set is returned.
    """
    if not hasattr(model, attribute):
        return list(index_set)
    variable = getattr(model, attribute)
    return [f for f in index_set if value(variable[f]) > 0.5]


@contextmanager
def _quiet_pyomo_solution_loading():
    """Silence Pyomo's warning about loading a solution from an aborted solve.

    A time limit is a supported outcome here and the result reports it through
    `is_optimal` and `gap`, so the warning would only alarm the caller without
    telling them anything the result does not already say.
    """
    logger = logging.getLogger("pyomo.core")
    previous_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(previous_level)


def _as_float(bound) -> float | None:
    if bound is None:
        return None
    try:
        return float(bound)
    except (TypeError, ValueError):
        return None


def _relative_gap(lower_bound: float | None, incumbent: float) -> float:
    """Relative MIP gap, the quantity solvers report.

    A proven-optimal result can still show a tiny non-zero gap: HiGHS stops once
    the gap falls under its default relative tolerance, so optimality is proven
    to within that tolerance rather than to the last decimal.
    """
    if lower_bound is None or incumbent == 0:
        return 0.0
    return abs(incumbent - lower_bound) / abs(incumbent)
