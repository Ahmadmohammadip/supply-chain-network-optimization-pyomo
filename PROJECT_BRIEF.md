# Multi-Echelon Supply Chain Network Optimization with Pyomo — Handoff Brief

## Purpose of this document
Handoff brief for Claude Code (or any engineer) to build this project from
scratch. It captures the locked scope, full mathematical formulation, repo
architecture, and phased build plan agreed on before any code was written.
No code exists yet — this is the starting specification. Fourth in a
series of companion portfolio repos (`economic-dispatch-pyomo`,
`battery-storage-optimization-pyomo`, `cvrp-optimization-pyomo`) — same
conventions and level of polish, but a fully independent, standalone repo
(no shared code).

## Goal
Public GitHub repo: a mixed-integer program (MILP) that jointly decides
(a) which plants and warehouses to open (strategic network design) and
(b) how to produce, ship, and hold inventory across a multi-period horizon
(operational planning) — a combined multi-echelon network design +
production-distribution-inventory model. Built and committed phase by
phase, each phase tested and working before the next. Public from commit 1.

## Scope (locked decisions)
- **Echelons**: plants → warehouses/DCs → customers. Three tiers, two
  shipping legs.
- **Strategic layer**: binary open/close decision for each candidate
  plant and each candidate warehouse. **A facility, once open, is
  considered open for the entire planning horizon** — no phased
  opening/closing over time. This is a deliberate simplification (see
  Section 5); modeling time-varying facility status would require
  time-indexed binaries and is out of scope here.
- **Operational layer**: multi-period production, shipping, and
  inventory-holding at warehouses, including a safety-stock requirement.
- **Product**: single product / single commodity. Multi-product is out
  of scope (see Section 5).
- **Demand**: deterministic and must be met exactly each period — no
  backorders, no stockouts, no lost sales. This means safety stock here
  is a **user-specified policy parameter**, not something statistically
  derived from demand variability (there is no variability in this
  deterministic model) — this must be stated plainly in
  `docs/formulation.md` so it isn't mistaken for a service-level-driven
  safety stock calculation.
- **Repo structure**: installable package (`src/scn_opt`) + notebook +
  Streamlit demo app, same shape as the other three repos.
- **Solver**: MILP (binary facility variables + continuous flow/inventory
  variables, linear objective and constraints) — HiGHS as the default
  free solver; CBC as a documented fallback. No Ipopt needed.

## 1. Mathematical Formulation

This follows the standard structure of multi-echelon facility-location /
network-design models widely used in the supply chain optimization
literature (see e.g. the review by Melo, Nickel & Saldanha-da-Gama,
*"Facility location and supply chain management – A review,"* European
Journal of Operational Research, 2009) — cite it as a well-established
class of model, not as an original formulation.

### 1.1 Sets and indices

| Symbol | Description |
|---|---|
| $P$ | Candidate plant locations, index $p$ |
| $W$ | Candidate warehouse/DC locations, index $w$ |
| $C$ | Customer (demand) locations, index $c$ |
| $T$ | Time periods, index $t = 1, \dots, T$ |

### 1.2 Parameters

| Symbol | Description |
|---|---|
| $f_p, f_w$ | Fixed cost to open plant $p$ / warehouse $w$ (one-time, incurred once if opened, amortized over the horizon) |
| $\overline{Q}_p$ | Production capacity of plant $p$ (units/period) |
| $\overline{Q}_w$ | Throughput/storage capacity of warehouse $w$ |
| $c^{prod}_p$ | Unit production cost at plant $p$ |
| $c^{PW}_{p,w}$ | Unit shipping cost, plant $p$ → warehouse $w$ |
| $c^{WC}_{w,c}$ | Unit shipping cost, warehouse $w$ → customer $c$ |
| $h_w$ | Unit inventory holding cost per period at warehouse $w$ |
| $ss_w$ | Safety stock requirement at warehouse $w$ (policy parameter, see Scope above) |
| $D_{c,t}$ | Demand of customer $c$ in period $t$ |
| $I_{w,0}$ | Initial inventory at warehouse $w$ (typically 0) |

### 1.3 Decision variables

| Symbol | Description |
|---|---|
| $y_p \in \{0,1\}$ | 1 if plant $p$ is opened (for the whole horizon) |
| $y_w \in \{0,1\}$ | 1 if warehouse $w$ is opened (for the whole horizon) |
| $x^{PW}_{p,w,t} \ge 0$ | Flow from plant $p$ to warehouse $w$ in period $t$ |
| $x^{WC}_{w,c,t} \ge 0$ | Flow from warehouse $w$ to customer $c$ in period $t$ |
| $I_{w,t} \ge 0$ | Inventory held at warehouse $w$ at the end of period $t$ |

### 1.4 Objective

$$
\min \sum_{p} f_p y_p + \sum_{w} f_w y_w + \sum_t \sum_{p} c^{prod}_p \sum_w x^{PW}_{p,w,t} + \sum_t \sum_{p,w} c^{PW}_{p,w} x^{PW}_{p,w,t} + \sum_t \sum_{w,c} c^{WC}_{w,c} x^{WC}_{w,c,t} + \sum_t \sum_w h_w I_{w,t}
$$

Fixed facility cost + production cost + inbound shipping + outbound
shipping + inventory holding.

### 1.5 Constraints

**Plant capacity** — a plant can only ship (in aggregate) up to its
capacity, and only if open:

$$
\sum_w x^{PW}_{p,w,t} \le \overline{Q}_p \, y_p \quad \forall p, t
$$

**Warehouse inventory balance**:

$$
I_{w,t} = I_{w,t-1} + \sum_p x^{PW}_{p,w,t} - \sum_c x^{WC}_{w,c,t} \quad \forall w, t
$$

**Warehouse flow and storage tied to open/closed status** — using
$\overline{Q}_w$ as a natural bound instead of an arbitrary big-M
constant (deliberate choice — avoids the numerical/tightness problems
generic big-M constraints can cause in a solver):

$$
\sum_p x^{PW}_{p,w,t} \le \overline{Q}_w \, y_w, \qquad \sum_c x^{WC}_{w,c,t} \le \overline{Q}_w \, y_w, \qquad I_{w,t} \le \overline{Q}_w \, y_w \quad \forall w, t
$$

**Safety stock** (only meaningful, and only binding, when the warehouse
is open — if $y_w = 0$ the constraints above already force $I_{w,t} = 0$):

$$
I_{w,t} \ge ss_w \, y_w \quad \forall w, t
$$

**Demand satisfaction** (exact — no backorders or stockouts modeled):

$$
\sum_w x^{WC}_{w,c,t} = D_{c,t} \quad \forall c, t
$$

**Domains**: $y_p, y_w \in \{0,1\}$; $x^{PW}, x^{WC}, I \ge 0$.

### 1.6 Feasibility note

An instance is infeasible if total demand in any period exceeds total
plant capacity, or if the candidate network simply cannot connect supply
to demand (e.g. a customer with no warehouse able to reach it). Basic
versions of these checks (aggregate capacity vs. aggregate demand) should
be validated at `System` construction time, same "fail loud, not at
solve time" philosophy as the other three repos — though note that
*network*-level feasibility (can flow actually route through the graph
given which facilities are candidates) is harder to check cheaply than
the simple sum-of-capacities check, and is reasonable to leave to the
solver's infeasibility report.

## 2. Repo architecture

```
supply-chain-network-optimization-pyomo/
├── README.md
├── LICENSE (MIT)
├── pyproject.toml
├── PROJECT_BRIEF.md
├── src/
│   └── scn_opt/
│       ├── __init__.py
│       ├── data/
│       │   ├── schema.py         # Plant, Warehouse, Customer, System (validated dataclasses)
│       │   └── loaders.py        # CSV/JSON -> validated data objects
│       ├── model/
│       │   ├── __init__.py
│       │   └── builder.py        # Pyomo ConcreteModel construction
│       ├── solve.py               # solver interface, result dataclass, extraction of open facilities/flows/inventory
│       └── viz.py                 # network diagram (open facilities + flow volumes), inventory trajectories, cost breakdown
├── data/
│   └── sample_networks/           # small synthetic instances (e.g. 2 plants / 3 warehouses / 5 customers / 4 periods)
├── notebooks/
│   └── 01_walkthrough.ipynb
├── app/
│   └── streamlit_app.py           # tune demand/capacity/costs, see which facilities open and resulting flows
├── tests/
│   ├── test_capacity_constraints.py
│   ├── test_inventory_balance.py
│   ├── test_facility_open_close_logic.py
│   └── test_integration.py
├── .github/workflows/ci.yml       # ruff + pytest, HiGHS only
└── docs/
    └── formulation.md             # Section 1 above, rendered, cited to the SCM literature appropriately
```

**Design rationale** (carried over from the other three repos):
`data/schema.py` validated dataclasses only — the model layer never
touches raw CSV/JSON. `System` should fail loudly at construction for
clear data errors (negative demand, aggregate demand exceeding aggregate
plant capacity even with everything open, a warehouse capacity of zero
with a positive safety stock requirement, etc.).

## 3. Build plan (phased)

| Phase | Scope | Output |
|---|---|---|
| 1 | Operational layer only: fixed network (all facilities pre-opened, no binaries, no fixed costs), multi-period flow + inventory balance + safety stock | Working LP — validates the time-coupling logic before adding combinatorial complexity |
| 2 | Add plant/warehouse binary open/close decisions + fixed costs → full MILP | Complete formulation from Section 1 |
| 3 | Data schema + loaders + `System` validation | `data/schema.py`, `data/loaders.py` |
| 4 | Sample network instances (small, hand-designed so the optimal network configuration is inspectable) | `data/sample_networks/` |
| 5 | `viz.py` — network diagram (open vs. closed facilities, flow volumes), inventory trajectory per warehouse, cost breakdown chart | Visual output |
| 6 | Notebook walkthrough | Narrated build → solve → visualize |
| 7 | Streamlit app | Interactive demo: tune demand/capacity/cost sliders, see which facilities open |
| 8 | Tests, CI, README, `docs/formulation.md` polish | GitHub-ready |

Each phase should leave `main` green (tests passing) before moving to the
next, and correspond to its own commit(s).

## 4. A worthwhile test case to include
A test where a warehouse with a **low fixed cost but high per-unit
shipping cost** loses to a warehouse with a **high fixed cost but low
per-unit shipping cost** once demand is large enough — i.e., the model
should visibly trade off strategic (fixed) vs. operational (variable)
cost as scale changes. This is the kind of test that actually
demonstrates the model does something non-trivial, similar in spirit to
the congestion/LMP-divergence test in `economic-dispatch-pyomo`.

## 5. Explicitly out of scope (do not build unless asked)
- Multi-product / multi-commodity flows
- Time-varying facility status (phased opening/closing within the horizon)
- Backorders, stockouts, or lost sales (demand must be met exactly)
- Statistically-derived safety stock (requires stochastic demand
  modeling — this repo's `ss_w` is a flat policy input, not computed)
- Transportation lead times (all flow is assumed to arrive within the
  same period it's shipped)
- Capacity expansion over time
- Economies of scale / quantity discounts on shipping or production cost
  (all unit costs are constant regardless of volume)

## 6. Git conventions
- One phase per commit (or a few if a phase is large), each commit
  leaves `main` green
- Commit message prefixes: `feat` / `test` / `docs` / `ci` / `chore` / `fix`
- Public repo from commit 1
- Suggested repo name: `supply-chain-network-optimization-pyomo`

## 7. Provenance note
This brief was authored directly in this conversation as a planning
document, before any code was written. The multi-echelon network design
+ inventory structure follows well-established supply chain optimization
literature (cited above), not an original formulation. Nothing in this
document should be treated as already implemented.
