"""
Streamlit demo for the supply chain network model.

Pick a sample network, scale demand, capacity, and the cost of opening
facilities, and watch which plants and warehouses the model keeps. Every run
also solves the open-everything counterfactual, so the value of the strategic
decision is on screen rather than implied.

Solving is cheap at these sizes, so it happens on every change rather than
behind a submit button — but results are cached, so moving a slider back to a
value you have already seen is instant.

Run with:  streamlit run app/streamlit_app.py
"""

import json
from pathlib import Path

import streamlit as st

from scn_opt.data.loaders import (
    load_demand_csv_text,
    load_system_json,
    system_from_dict,
    system_to_dict,
)
from scn_opt.model.builder import build_from_system
from scn_opt.solve import solve_scn
from scn_opt.viz import plot_cost_breakdown, plot_inventory_trajectories, plot_network

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_networks"

st.set_page_config(page_title="Supply Chain Network Optimization", layout="wide")

st.title("Multi-echelon supply chain network design")
st.caption(
    "One MILP choosing which plants and warehouses to open and how to produce, ship, "
    "and hold inventory across the horizon — solved together, because choosing the "
    "network first means guessing at the operating costs it implies."
)


@st.cache_data(show_spinner=False)
def solve_network(payload: str, force_open_all: bool):
    """Cached solve, keyed on the instance JSON so identical settings are free."""
    system = system_from_dict(json.loads(payload))
    return system, solve_scn(build_from_system(system, force_open_all=force_open_all))


def scaled_payload(
    system, demand_scale: float, capacity_scale: float, fixed_cost_scale: float
) -> str:
    """Apply the sliders to a base network and return it as a JSON payload.

    Scaling a committed instance rather than exposing every parameter keeps the
    controls to the three levers that actually change the answer.
    """
    data = system_to_dict(system)
    for plant in data["plants"]:
        plant["capacity"] *= capacity_scale
        plant["fixed_cost"] *= fixed_cost_scale
    for warehouse in data["warehouses"]:
        warehouse["fixed_cost"] *= fixed_cost_scale
    for customer in data["customers"]:
        customer["demand"] = [d * demand_scale for d in customer["demand"]]
    return json.dumps(data)


sample_files = sorted(SAMPLE_DIR.glob("*.json"))

with st.sidebar:
    st.header("Network")
    choice = st.selectbox("Sample network", sample_files, format_func=lambda p: p.stem)
    base_system = load_system_json(choice)

    st.caption(
        f"{len(base_system.plants)} candidate plants · "
        f"{len(base_system.warehouses)} candidate warehouses · "
        f"{len(base_system.customers)} customers · {base_system.n_periods} periods"
    )

    st.header("Scenario")
    demand_scale = st.slider(
        "Demand", 0.2, 3.0, 1.0, step=0.1,
        help="Scales every customer's demand. Volume is what tips the fixed-versus-"
             "variable balance.",
    )
    capacity_scale = st.slider(
        "Plant capacity", 0.5, 3.0, 1.0, step=0.1,
        help="Scales every plant's capacity per period.",
    )
    fixed_cost_scale = st.slider(
        "Cost of opening facilities", 0.0, 3.0, 1.0, step=0.1,
        help="Scales every fixed cost. At zero, opening is free and the model keeps "
             "everything it finds useful.",
    )

    st.header("Demand override")
    uploaded = st.file_uploader(
        "CSV with columns `customer`, `period`, `demand` — replaces the demand of "
        "customers whose names match",
        type="csv",
    )

payload = scaled_payload(base_system, demand_scale, capacity_scale, fixed_cost_scale)

if uploaded is not None:
    try:
        replacements = {
            c.name: c.demand
            for c in load_demand_csv_text(
                uploaded.getvalue().decode("utf-8"), label=uploaded.name
            )
        }
    except ValueError as exc:
        st.error(f"Could not read {uploaded.name}: {exc}")
        st.stop()

    data = json.loads(payload)
    known = {c["name"] for c in data["customers"]}
    unknown = set(replacements) - known
    if unknown:
        st.warning(
            f"Ignoring {sorted(unknown)} — not customers of the {choice.stem} network. "
            f"It expects: {sorted(known)}."
        )
    for customer in data["customers"]:
        if customer["name"] in replacements:
            customer["demand"] = replacements[customer["name"]]
    payload = json.dumps(data)
    st.info(f"Demand for {sorted(known & set(replacements))} taken from {uploaded.name}.")

try:
    system, result = solve_network(payload, False)
    _, everything = solve_network(payload, True)
except ValueError as exc:
    # The schema rejects instances the solver could only report as "infeasible",
    # so this is the useful message, not a fallback.
    st.error(f"This scenario is not solvable: {exc}")
    st.stop()
except RuntimeError as exc:
    st.error(f"{exc}")
    st.stop()

saving = everything.total_cost - result.total_cost

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total cost", f"{result.total_cost:,.0f}")
col2.metric(
    "Plants open", f"{len(result.open_plants)} of {len(system.plants)}"
)
col3.metric(
    "Warehouses open", f"{len(result.open_warehouses)} of {len(system.warehouses)}"
)
col4.metric(
    "vs. opening everything",
    f"{everything.total_cost:,.0f}",
    delta=f"-{saving:,.0f}" if saving > 0 else "0",
    delta_color="inverse",
)

if saving > 0:
    st.success(
        f"Choosing the network saves {saving:,.0f} ({saving / everything.total_cost:.1%}) "
        f"against running every candidate facility. Note where it comes from below — "
        f"closing sites cuts fixed cost but lengthens hauls, so shipping usually rises."
    )
else:
    st.info(
        "At this scenario every candidate facility earns its keep, so the optimized "
        "network is the whole network."
    )

st.pyplot(plot_network(system, result))

left, right = st.columns(2)
with left:
    st.pyplot(plot_cost_breakdown(system, result, baseline=everything))
with right:
    st.pyplot(plot_inventory_trajectories(system, result))

with st.expander("Which facilities opened, and what each carries"):
    for plant in system.plants:
        produced = sum(
            q for (p, _, _), q in result.flows_plant_to_warehouse.items()
            if p == plant.name
        )
        state = "open" if plant.name in result.open_plants else "closed"
        st.markdown(
            f"**{plant.name}** ({state}) — capacity {plant.capacity:,.0f}/period, "
            f"produced {produced:,.0f} over the horizon"
        )
    for warehouse in system.warehouses:
        shipped = sum(
            q for (w, _, _), q in result.flows_warehouse_to_customer.items()
            if w == warehouse.name
        )
        state = "open" if warehouse.name in result.open_warehouses else "closed"
        st.markdown(
            f"**{warehouse.name}** ({state}) — throughput "
            f"{warehouse.throughput_capacity:,.0f}/period, storage "
            f"{warehouse.storage_capacity:,.0f}, shipped {shipped:,.0f} over the horizon"
        )

with st.expander("What this model does not do"):
    st.markdown(
        """
- **Single product.** No multi-commodity flows.
- **Facilities open for the whole horizon.** No phased opening or closing, which
  would need time-indexed binaries.
- **Demand met exactly.** No backorders, stockouts, or lost sales — an
  unservable scenario is infeasible rather than merely expensive.
- **Safety stock is a flat policy input**, not derived from demand variability;
  there is none in this deterministic model to derive it from. It is also never
  drawn down, so it is produced once and charged holding cost every period.
- **No lead times.** Everything shipped arrives in the period it leaves.
- **Constant unit costs.** No economies of scale or quantity discounts.

Full list with rationale in `docs/formulation.md`.
"""
    )
