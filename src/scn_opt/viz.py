"""
Plotting helpers for a solved network plan.

matplotlib only, kept dependency-light so these work headless in CI and inside
the Streamlit app. Each function takes a `System` and a `NetworkResult` and
returns a Figure — callers decide whether to show(), save(), or hand it to
st.pyplot().
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from scn_opt.data.schema import System
from scn_opt.solve import NetworkResult

OPEN_PLANT_COLOR = "#1f77b4"
OPEN_WAREHOUSE_COLOR = "#2ca02c"
CUSTOMER_COLOR = "#4b4b4b"
CLOSED_COLOR = "#bdbdbd"
FLOW_COLOR = "#5a5a5a"

# Line widths for the thinnest and thickest lane on the diagram. Flow volumes
# are scaled between these rather than used directly, so one enormous lane
# cannot render every other lane invisible.
MIN_LINE_WIDTH = 0.6
MAX_LINE_WIDTH = 6.0


def _column_positions(names: list, x: float) -> dict:
    """Evenly spaced vertical positions for one echelon, centred on zero."""
    count = len(names)
    if count == 1:
        return {names[0]: (x, 0.0)}
    span = max(count - 1, 1)
    return {
        name: (x, 1.0 - 2.0 * index / span) for index, name in enumerate(names)
    }


def _line_width(volume: float, largest: float) -> float:
    if largest <= 0:
        return MIN_LINE_WIDTH
    return MIN_LINE_WIDTH + (MAX_LINE_WIDTH - MIN_LINE_WIDTH) * (volume / largest)


def plot_network(system: System, result: NetworkResult):
    """The chosen network: which facilities opened, and how much moves on each lane.

    Three columns — plants, warehouses, customers — with lane thickness set by
    total volume over the whole horizon. Facilities that stayed closed are drawn
    hollow and grey, because the ones that were available but rejected are as
    much a part of the answer as the ones that opened.
    """
    plant_names = [p.name for p in system.plants]
    warehouse_names = [w.name for w in system.warehouses]
    customer_names = [c.name for c in system.customers]

    plant_pos = _column_positions(plant_names, 0.0)
    warehouse_pos = _column_positions(warehouse_names, 1.0)
    customer_pos = _column_positions(customer_names, 2.0)

    # Aggregate over periods: the diagram answers "what does this network carry",
    # not "what happens in period 3".
    inbound_volume: dict = {}
    for (p, w, _), qty in result.flows_plant_to_warehouse.items():
        inbound_volume[(p, w)] = inbound_volume.get((p, w), 0.0) + qty
    outbound_volume: dict = {}
    for (w, c, _), qty in result.flows_warehouse_to_customer.items():
        outbound_volume[(w, c)] = outbound_volume.get((w, c), 0.0) + qty

    largest = max([*inbound_volume.values(), *outbound_volume.values(), 0.0])

    fig, ax = plt.subplots(figsize=(11, 7))

    for (p, w), volume in inbound_volume.items():
        (x1, y1), (x2, y2) = plant_pos[p], warehouse_pos[w]
        ax.plot(
            [x1, x2], [y1, y2],
            color=FLOW_COLOR, alpha=0.55, zorder=1,
            linewidth=_line_width(volume, largest),
        )
    for (w, c), volume in outbound_volume.items():
        (x1, y1), (x2, y2) = warehouse_pos[w], customer_pos[c]
        ax.plot(
            [x1, x2], [y1, y2],
            color=FLOW_COLOR, alpha=0.55, zorder=1,
            linewidth=_line_width(volume, largest),
        )

    open_plants = set(result.open_plants)
    open_warehouses = set(result.open_warehouses)

    def _draw(positions: dict, is_open, color: str, marker: str, size: float):
        for name, (x, y) in positions.items():
            opened = is_open(name)
            ax.scatter(
                [x], [y],
                s=size, marker=marker, zorder=3,
                color=color if opened else "white",
                edgecolor=color if opened else CLOSED_COLOR,
                linewidth=1.8,
            )
            label = name if opened else f"{name}\n(closed)"
            ax.annotate(
                label, (x, y),
                # Far enough below to clear the marker: the two-line "(closed)"
                # label would otherwise sit on top of its own symbol.
                textcoords="offset points", xytext=(0, -28),
                ha="center", va="top", fontsize=8,
                color="black" if opened else CLOSED_COLOR,
            )

    _draw(plant_pos, lambda n: n in open_plants, OPEN_PLANT_COLOR, "s", 320)
    _draw(warehouse_pos, lambda n: n in open_warehouses, OPEN_WAREHOUSE_COLOR, "o", 320)
    _draw(customer_pos, lambda n: True, CUSTOMER_COLOR, "^", 200)

    for x, label in ((0.0, "Plants"), (1.0, "Warehouses"), (2.0, "Customers")):
        ax.annotate(
            label, (x, 1.35), ha="center", fontsize=11, fontweight="bold"
        )

    ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(-1.5, 1.55)
    ax.axis("off")
    ax.set_title(
        f"Network plan — {result.total_cost:,.0f} total cost\n"
        f"{len(open_plants)}/{len(plant_names)} plants and "
        f"{len(open_warehouses)}/{len(warehouse_names)} warehouses open; "
        f"lane thickness is volume over the horizon"
    )
    fig.tight_layout()
    return fig


def plot_inventory_trajectories(system: System, result: NetworkResult):
    """Stock held at each warehouse over the horizon.

    The safety-stock floor is drawn as a dashed line. A trajectory that sits
    flat on its floor is a warehouse carrying nothing but its policy buffer —
    stock that is produced once and then charged holding cost forever, never
    drawn down.
    """
    periods = system.periods
    open_warehouses = set(result.open_warehouses)

    fig, ax = plt.subplots(figsize=(10, 5))

    plotted = False
    for index, warehouse in enumerate(system.warehouses):
        if warehouse.name not in open_warehouses:
            continue
        plotted = True
        color = f"C{index % 10}"
        levels = [result.inventory[(warehouse.name, t)] for t in periods]
        ax.plot(
            [0, *periods],
            [warehouse.initial_inventory, *levels],
            marker="o", markersize=4, color=color, label=warehouse.name,
        )
        if warehouse.safety_stock > 0:
            ax.axhline(
                warehouse.safety_stock,
                color=color, linestyle="--", linewidth=1, alpha=0.7,
            )
            ax.annotate(
                f"{warehouse.name} safety stock",
                (0, warehouse.safety_stock),
                textcoords="offset points", xytext=(4, 4),
                fontsize=7, color=color,
            )

    if not plotted:
        ax.annotate(
            "no warehouses open", (0.5, 0.5),
            xycoords="axes fraction", ha="center", color=CLOSED_COLOR,
        )

    ax.set_xlabel("Period")
    ax.set_ylabel("Units held")
    ax.set_xticks([0, *periods])
    ax.set_title("Inventory held at each open warehouse")
    if plotted:
        ax.legend(loc="best", fontsize="small")
    ax.margins(y=0.15)
    fig.tight_layout()
    return fig


def plot_cost_breakdown(
    system: System, result: NetworkResult, baseline: NetworkResult | None = None
):
    """Where the money goes, by cost component.

    Pass `baseline` — the same network solved with every facility forced open —
    to show the two side by side. The gap between them is what the strategic
    decision is worth, which is easy to assert and easy to miss.
    """
    breakdown = result.cost_breakdown()
    labels = list(breakdown)
    values = [breakdown[k] for k in labels]
    pretty = [k.replace("_", " ") for k in labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    positions = range(len(labels))

    if baseline is None:
        ax.bar(positions, values, color="#1f77b4")
        ax.set_xticks(list(positions))
        ax.set_xticklabels(pretty, rotation=15, ha="right")
        for x, value in zip(positions, values, strict=True):
            ax.annotate(
                f"{value:,.0f}", (x, value), ha="center", va="bottom", fontsize=8
            )
        ax.set_title(f"Cost breakdown — {result.total_cost:,.0f} total")
    else:
        baseline_values = [baseline.cost_breakdown()[k] for k in labels]
        width = 0.38
        ax.bar(
            [x - width / 2 for x in positions], values, width,
            color="#1f77b4", label=f"Optimized — {result.total_cost:,.0f}",
        )
        ax.bar(
            [x + width / 2 for x in positions], baseline_values, width,
            color="#bdbdbd", label=f"Every facility open — {baseline.total_cost:,.0f}",
        )
        ax.set_xticks(list(positions))
        ax.set_xticklabels(pretty, rotation=15, ha="right")
        ax.legend(loc="best", fontsize="small")
        saving = baseline.total_cost - result.total_cost
        ax.set_title(
            f"Cost breakdown — choosing the network saves {saving:,.0f} "
            f"({saving / baseline.total_cost:.1%})"
        )

    ax.set_ylabel("Cost")
    ax.margins(y=0.15)
    fig.tight_layout()
    return fig
