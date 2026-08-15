"""Load validated instances from JSON, and demand tables from CSV.

A network is nested — facilities, customers, and the costs of the arcs between
them — so a whole instance lives in JSON. Demand is flat and time-indexed,
which is what CSV is good at, so a demand table can be loaded on its own and
dropped into an existing network. That is the shape of the common workflow:
the network changes rarely, the forecast changes constantly.

Everything returned here has already passed the schema's validation, so a
malformed file fails at load rather than after a solve.
"""

import csv
import json
from pathlib import Path

from scn_opt.data.schema import Customer, Plant, System, Warehouse

CUSTOMER_COLUMN = "customer"
PERIOD_COLUMN = "period"
DEMAND_COLUMN = "demand"
REQUIRED_DEMAND_COLUMNS = (CUSTOMER_COLUMN, PERIOD_COLUMN, DEMAND_COLUMN)


def load_system_json(path: str | Path) -> System:
    """Read a complete instance from JSON.

    Expected shape — arc costs are lists of records because JSON has no tuple
    keys:

        {
          "plants": [
            {"name": "P1", "capacity": 100, "production_cost": 1, "fixed_cost": 500}
          ],
          "warehouses": [
            {"name": "W1", "throughput_capacity": 80, "storage_capacity": 40,
             "fixed_cost": 200, "holding_cost": 0.5, "safety_stock": 10}
          ],
          "customers": [{"name": "C1", "demand": [10, 20, 30, 15]}],
          "cost_plant_to_warehouse": [{"plant": "P1", "warehouse": "W1", "cost": 2}],
          "cost_warehouse_to_customer": [{"warehouse": "W1", "customer": "C1", "cost": 3}]
        }
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return system_from_dict(data, label=path.name)


def system_from_dict(data: dict, label: str = "instance") -> System:
    """Build a `System` from an already-parsed instance dict."""
    required = (
        "plants",
        "warehouses",
        "customers",
        "cost_plant_to_warehouse",
        "cost_warehouse_to_customer",
    )
    for key in required:
        if key not in data:
            raise ValueError(f"{label}: missing required key {key!r}")

    plants = [Plant(**p) for p in data["plants"]]
    warehouses = [Warehouse(**w) for w in data["warehouses"]]
    customers = [Customer(**c) for c in data["customers"]]

    inbound = _arc_costs(
        data["cost_plant_to_warehouse"], "plant", "warehouse", label,
        "cost_plant_to_warehouse",
    )
    outbound = _arc_costs(
        data["cost_warehouse_to_customer"], "warehouse", "customer", label,
        "cost_warehouse_to_customer",
    )

    return System(
        plants=plants,
        warehouses=warehouses,
        customers=customers,
        cost_plant_to_warehouse=inbound,
        cost_warehouse_to_customer=outbound,
    )


def system_to_dict(system: System) -> dict:
    """Inverse of `system_from_dict`, for writing instances back out.

    Used by the sample-instance generator and by the app, which round-trips a
    system through this shape so an uploaded demand table and a bundled
    instance take exactly the same path into the model.
    """
    return {
        "plants": [
            {
                "name": p.name,
                "capacity": p.capacity,
                "production_cost": p.production_cost,
                "fixed_cost": p.fixed_cost,
            }
            for p in system.plants
        ],
        "warehouses": [
            {
                "name": w.name,
                "throughput_capacity": w.throughput_capacity,
                "storage_capacity": w.storage_capacity,
                "fixed_cost": w.fixed_cost,
                "holding_cost": w.holding_cost,
                "safety_stock": w.safety_stock,
                "initial_inventory": w.initial_inventory,
            }
            for w in system.warehouses
        ],
        "customers": [{"name": c.name, "demand": list(c.demand)} for c in system.customers],
        "cost_plant_to_warehouse": [
            {"plant": p, "warehouse": w, "cost": cost}
            for (p, w), cost in system.cost_plant_to_warehouse.items()
        ],
        "cost_warehouse_to_customer": [
            {"warehouse": w, "customer": c, "cost": cost}
            for (w, c), cost in system.cost_warehouse_to_customer.items()
        ],
    }


def load_demand_csv(path: str | Path) -> list[Customer]:
    """Read a demand table in long form: one row per customer per period.

    Requires columns `customer`, `period`, `demand`. Periods are 1-based and
    every customer must cover the same set of them.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return _customers_from_rows(rows, path.name)


def load_demand_csv_text(text: str, label: str = "uploaded.csv") -> list[Customer]:
    """Same as `load_demand_csv` for CSV already in memory — an uploaded file in
    the Streamlit app. `label` only appears in error messages."""
    rows = list(csv.DictReader(text.lstrip("﻿").splitlines()))
    return _customers_from_rows(rows, label)


def _customers_from_rows(rows: list[dict], label: str) -> list[Customer]:
    if not rows:
        raise ValueError(f"{label}: file has a header but no data rows")

    columns = rows[0].keys()
    missing = [c for c in REQUIRED_DEMAND_COLUMNS if c not in columns]
    if missing:
        raise ValueError(
            f"{label}: missing required column(s) {missing} "
            f"(found: {sorted(c for c in columns if c)})"
        )

    by_customer: dict[str, dict[int, float]] = {}
    for line_number, row in enumerate(rows, start=2):  # row 1 is the header
        name = (row[CUSTOMER_COLUMN] or "").strip()
        if not name:
            raise ValueError(f"{label} line {line_number}: empty customer name")
        period = _parse_int(row[PERIOD_COLUMN], PERIOD_COLUMN, label, line_number)
        demand = _parse_float(row[DEMAND_COLUMN], DEMAND_COLUMN, label, line_number)

        if period < 1:
            raise ValueError(
                f"{label} line {line_number}: period must be >= 1, got {period}"
            )
        periods = by_customer.setdefault(name, {})
        if period in periods:
            raise ValueError(
                f"{label} line {line_number}: customer {name!r} has two rows for "
                f"period {period}"
            )
        periods[period] = demand

    all_periods = {frozenset(p) for p in by_customer.values()}
    if len(all_periods) != 1:
        raise ValueError(
            f"{label}: every customer must cover the same periods, but they differ — "
            f"check for a missing or extra row"
        )

    period_numbers = sorted(next(iter(all_periods)))
    expected = list(range(1, len(period_numbers) + 1))
    if period_numbers != expected:
        raise ValueError(
            f"{label}: periods must run 1..{len(period_numbers)} with no gaps, "
            f"got {period_numbers}"
        )

    return [
        Customer(name=name, demand=[periods[t] for t in period_numbers])
        for name, periods in by_customer.items()
    ]


def _arc_costs(
    records: list, from_key: str, to_key: str, label: str, field_name: str
) -> dict:
    costs = {}
    for index, record in enumerate(records):
        for key in (from_key, to_key, "cost"):
            if key not in record:
                raise ValueError(
                    f"{label}: {field_name}[{index}] is missing {key!r}"
                )
        pair = (record[from_key], record[to_key])
        if pair in costs:
            raise ValueError(f"{label}: {field_name} has two entries for {pair}")
        costs[pair] = float(record["cost"])
    return costs


def _parse_int(raw: str, column: str, label: str, line_number: int) -> int:
    value = (raw or "").strip()
    if not value:
        raise ValueError(f"{label} line {line_number}: empty value in column {column!r}")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"{label} line {line_number}: column {column!r} is not a whole number ({raw!r})"
        ) from exc


def _parse_float(raw: str, column: str, label: str, line_number: int) -> float:
    value = (raw or "").strip()
    if not value:
        raise ValueError(f"{label} line {line_number}: empty value in column {column!r}")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"{label} line {line_number}: column {column!r} is not a number ({raw!r})"
        ) from exc
