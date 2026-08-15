# Multi-Echelon Supply Chain Network Optimization with Pyomo

A mixed-integer program that makes two kinds of decision **in one solve**:

- **Strategic** — which plants and warehouses to open, paid for once, for the
  whole horizon.
- **Operational** — how much to produce, ship on each leg, and hold as
  inventory, period by period.

Plants → warehouses → customers, single product, deterministic demand met
exactly. Built with [Pyomo](https://www.pyomo.org/), solved with HiGHS.

Solving the two together is the point. Choose the network first and you are
guessing at the operating costs it implies; optimize the flows first and the
network is already fixed.

> Built incrementally, phase by phase — see `PROJECT_BRIEF.md` for the scope
> agreed before any code was written, and `docs/formulation.md` for the full
> formulation, citations, and assumptions.

## What the trade-off looks like

On the `baseline` sample network — 2 candidate plants, 3 candidate warehouses,
5 customers, 4 periods — the model opens 1 plant and 2 warehouses, rejecting the
central warehouse and the northern plant.

Against the counterfactual of running every candidate facility:

| Component | Optimized | Every facility open | Change |
|---|---:|---:|---:|
| Fixed | 1,100 | 2,300 | **−1,200** |
| Production | 522 | 475 | +47 |
| Inbound shipping | 1,062.5 | 690 | **+372.5** |
| Outbound shipping | 692.5 | 500 | **+192.5** |
| Holding | 12 | 20 | −8 |
| **Total** | **3,389** | **3,985** | **−596** |

The saving is not from doing less of everything. Closing facilities cuts fixed
cost by 1,200 but adds 565 of shipping, because a smaller network means longer
hauls. That trade is the whole reason the two layers are solved together, and a
single total would have hidden it.

Scale moves the answer in both directions: at double demand every candidate
facility earns its keep and the optimized network *is* the whole network.

## Features

- **One MILP, both layers** — facility binaries and multi-period flows in a
  single model, not a network chosen and then operated.
- **Capacity instead of big-M** — every open/close binary multiplies a real
  capacity. A loose big-M weakens the relaxation and invites numerical trouble;
  the capacity is both the tightest valid bound and the meaningful one.
- **Throughput and storage as separate limits** — a warehouse is rated for what
  it can move per period and, separately, for what it can keep. That makes a
  cross-dock expressible and is what the brief's single parameter could not do.
- **An honest counterfactual** — `force_open_all` pins every facility open while
  still charging fixed costs, so the value of the strategic decision is measured
  rather than asserted.
- **Validated instances** — a `System` fails at construction on duplicate names,
  mismatched demand horizons, missing lane costs, or demand beyond what the
  candidate network could ever produce or ship.

## Two departures from the brief

Both recorded in `PROJECT_BRIEF.md` §8, both with the reasoning.

**Warehouse capacity is split in two** (§8.1). The brief's §1.5 uses one
parameter for inbound, outbound, and storage at once. Splitting it costs one
schema field and makes a cross-dock — high throughput, no storage —
expressible. Leaving `storage_capacity` unset recovers the original behavior.

**The feasibility check is cumulative, not per period** (§8.2). The brief's §1.6
calls an instance infeasible if demand in any period exceeds total plant
capacity. That is too strict for a model that holds inventory: production can be
banked, so a peak larger than one period of output is servable by building
ahead. Applied literally, that rule rejects the `spike` sample network, which
solves perfectly well. What holds instead is cumulative production, plus a
genuinely per-period check on warehouse outbound throughput — the one limit
pre-building cannot relax.

## Install

```bash
pip install -e ".[dev,solvers,viz]"
```

[HiGHS](https://highs.dev/) arrives with the `solvers` extra (`highspy`) and is
the only solver needed. [CBC](https://github.com/coin-or/Cbc) works as a
fallback if the binary is on your PATH: pass `solver_name="cbc"` to `solve_scn`.

## Quickstart

```python
from scn_opt.data.loaders import load_system_json
from scn_opt.model.builder import build_from_system
from scn_opt.solve import solve_scn

system = load_system_json("data/sample_networks/baseline.json")
result = solve_scn(build_from_system(system))

print(result.summary())
# 3,389.00 total cost — 1 plant(s), 2 warehouse(s) — proven optimal, 0.1s

print(result.open_warehouses)     # ['W_east', 'W_west']
print(result.cost_breakdown())    # sums to total_cost by construction

# What was the strategic decision worth?
everything = solve_scn(build_from_system(system, force_open_all=True))
print(everything.total_cost - result.total_cost)   # 596.0
```

## Repo layout

```
supply-chain-network-optimization-pyomo/
├── src/scn_opt/
│   ├── data/schema.py      # Plant, Warehouse, Customer, System (validated)
│   ├── data/loaders.py     # JSON networks, CSV demand tables
│   ├── model/builder.py    # Pyomo ConcreteModel construction
│   ├── solve.py            # solver interface, NetworkResult, cost breakdown
│   └── viz.py              # network diagram, inventory, cost breakdown
├── data/sample_networks/   # synthetic baseline / tradeoff / spike instances
├── notebooks/01_walkthrough.ipynb
├── app/streamlit_app.py
├── tests/
├── docs/formulation.md
└── .github/workflows/ci.yml
```

## Interactive demo

```bash
streamlit run app/streamlit_app.py
```

Scale demand, plant capacity, and the cost of opening facilities, and watch
which sites the model keeps. Every run also solves the open-everything
counterfactual, so the value of choosing is on screen. Demand can be overridden
from a CSV.

## Tests

```bash
pytest -v
```

The sample networks are small by design, so the whole suite — including every
sample instance — runs in a few seconds.

## A note on the sample data

The networks in `data/sample_networks/` are **synthetic and hand-designed**, not
drawn from any real supply chain. Costs are round numbers chosen to make a
particular trade-off visible. They are small enough that the optimal
configuration can be reasoned about rather than taken on trust, which is the
point of them. The generator is committed alongside.

## Known simplifications

Single product, facilities open for the whole horizon (no phased
opening/closing), demand met exactly with no backorders, no transport lead
times, no capacity expansion, and constant unit costs with no economies of
scale. Safety stock is a **flat policy input, not a service-level calculation** —
and it is never drawn down, so it is produced once and charged holding cost in
every period. Full list with rationale in `docs/formulation.md`.

## Companion repos

Four standalone optimization models built to the same conventions — validated
dataclasses that fail loudly at construction, a Pyomo builder that never touches
raw files, and a result dataclass rather than a live model — but sharing no code.

- [economic-dispatch-pyomo](https://github.com/Ahmadmohammadip/economic-dispatch-pyomo)
  — multi-period, multi-bus DC-OPF economic dispatch with generator ramping,
  curtailable renewables, storage, and locational marginal prices.
- [battery-storage-optimization-pyomo](https://github.com/Ahmadmohammadip/battery-storage-optimization-pyomo)
  — battery energy arbitrage co-optimized with frequency regulation capacity
  (revenue stacking) as a single LP.
- [cvrp-optimization-pyomo](https://github.com/Ahmadmohammadip/cvrp-optimization-pyomo)
  — exact MILP for the Capacitated Vehicle Routing Problem, with a measured
  benchmark of where exact methods stop scaling.

## License

MIT — see `LICENSE`.
