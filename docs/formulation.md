# Formulation

A mixed-integer linear program that decides, in one solve, which facilities to
open and how to run them: plants → warehouses → customers, over a multi-period
horizon, meeting deterministic demand exactly.

## Provenance

This follows the standard structure of **multi-echelon facility-location and
network-design models** used widely in the supply chain optimization
literature. It is a well-established class of model, not an original
formulation.

For a survey of the field and this family of models, see Melo, Nickel &
Saldanha-da-Gama, "Facility location and supply chain management — A review",
*European Journal of Operational Research* 196(2), 2009.

## 1. Sets and indices

| Symbol | Description | Code |
|---|---|---|
| $P$ | Candidate plants | `m.P` |
| $W$ | Candidate warehouses | `m.W` |
| $C$ | Customers | `m.C` |
| $T$ | Periods, $t = 1 \dots T$ | `m.T` |

## 2. Parameters

| Symbol | Description | Code |
|---|---|---|
| $f_p, f_w$ | Cost to open a plant / warehouse, charged once | `fixed_cost_plant`, `fixed_cost_warehouse` |
| $\overline{Q}_p$ | Plant production capacity per period | `plant_capacity` |
| $\overline{U}_w$ | Warehouse throughput capacity per period | `throughput_capacity` |
| $\overline{S}_w$ | Warehouse storage capacity | `storage_capacity` |
| $c^{prod}_p$ | Unit production cost | `production_cost` |
| $c^{PW}_{p,w}, c^{WC}_{w,c}$ | Unit shipping cost on each leg | `cost_pw`, `cost_wc` |
| $h_w$ | Unit holding cost per period | `holding_cost` |
| $ss_w$ | Safety stock floor (policy input — see §6) | `safety_stock` |
| $D_{c,t}$ | Customer demand | `demand` |
| $I_{w,0}$ | Opening inventory | `initial_inventory` |

## 3. Decision variables

| Symbol | Description | Code |
|---|---|---|
| $y_p, y_w \in \{0,1\}$ | 1 if the facility is opened, for the whole horizon | `open_plant`, `open_warehouse` |
| $x^{PW}_{p,w,t} \ge 0$ | Flow plant → warehouse | `ship_pw` |
| $x^{WC}_{w,c,t} \ge 0$ | Flow warehouse → customer | `ship_wc` |
| $I_{w,t} \ge 0$ | Inventory held at end of period | `inventory` |

## 4. Objective

$$
\min \; \underbrace{\sum_p f_p y_p + \sum_w f_w y_w}_{\text{fixed}}
\;+\; \sum_t \sum_{p,w} \left( c^{prod}_p + c^{PW}_{p,w} \right) x^{PW}_{p,w,t}
\;+\; \sum_t \sum_{w,c} c^{WC}_{w,c} x^{WC}_{w,c,t}
\;+\; \sum_t \sum_w h_w I_{w,t}
$$

Fixed facility cost, production, inbound shipping, outbound shipping, holding.
`solve.py` reports these five components separately, taken from the same
expressions the objective is built from, so the breakdown sums to the total by
construction rather than by a second calculation that might disagree.

## 5. Constraints

### 5.1 Plant capacity

$$
\sum_w x^{PW}_{p,w,t} \le \overline{Q}_p \, y_p \quad \forall p, t
$$

### 5.2 Inventory balance

$$
I_{w,t} = I_{w,t-1} + \sum_p x^{PW}_{p,w,t} - \sum_c x^{WC}_{w,c,t} \quad \forall w, t
$$

with $I_{w,0}$ given. Everything shipped arrives in the period it leaves —
transit lead times are out of scope.

### 5.3 Throughput and storage — two capacities, not one

$$
\sum_p x^{PW}_{p,w,t} \le \overline{U}_w \, y_w, \qquad
\sum_c x^{WC}_{w,c,t} \le \overline{U}_w \, y_w, \qquad
I_{w,t} \le \overline{S}_w \, y_w \quad \forall w, t
$$

**This departs from `PROJECT_BRIEF.md` §1.5**, which uses a single parameter
$\overline{Q}_w$ for all three limits. See §8.1 of the brief for the amendment.

A warehouse is rated for two different things: how much it can *move* in a
period, and how much it can *keep* between them. Collapsing them means a site
rated 100 may receive 100, ship 100, and hold 100 simultaneously, and makes a
cross-dock — high throughput, no storage — inexpressible. Splitting costs one
schema field. Leaving `storage_capacity` unset makes it equal throughput, which
recovers the brief's behavior exactly.

The property the brief cares about is untouched: each binary multiplies a real
capacity, never an arbitrary big-M constant. That matters because a loose big-M
weakens the linear relaxation and invites numerical trouble; the capacity is
both the tightest valid bound and the physically meaningful one.

### 5.4 Safety stock

$$
I_{w,t} \ge ss_w \, y_w \quad \forall w, t
$$

Gated on the binary so it applies only to warehouses that open — a closed one is
already forced to zero inventory by §5.3.

### 5.5 Demand met exactly

$$
\sum_w x^{WC}_{w,c,t} = D_{c,t} \quad \forall c, t
$$

Equality, not $\ge$: no backorders, no stockouts, no lost sales. An instance
that cannot be served is infeasible rather than expensive.

## 6. Safety stock is a policy input, and it is never consumed

Two things to be clear about, because both are easy to misread.

**It is not statistically derived.** Real safety stock is calculated from demand
variability and a target service level. This model's demand is deterministic —
there is no variability to derive anything from. $ss_w$ is a flat number the
user supplies, and nothing more.

**It is never drawn down.** The constraint requires the floor to hold in *every*
period, so stock held to satisfy it is produced once and then sits there. It is
not a reserve the plan dips into during a peak. On the baseline sample network
this is visible directly: 425 units of demand, 435 units produced, the
difference being buffer that is never sold and pays holding cost in all four
periods.

If what you want is a buffer that can be consumed under stress, this is not it,
and modeling it would require stochastic demand — which is out of scope.

## 7. Facilities open for the whole horizon

$y_p$ and $y_w$ carry no time index. A facility is open for the entire horizon
or not at all; phased opening and closing would need time-indexed binaries and
is out of scope (`PROJECT_BRIEF.md` §5).

A consequence worth knowing: **fixed cost is charged once, not per period**, so
horizon length is itself a lever on the strategic/operational balance. A longer
horizon accumulates more variable cost against the same one-off fixed cost,
making the model readier to pay for a facility that ships cheaply. The test
suite pins this: holding demand constant and stretching the horizon from 4
periods to 40 flips which warehouse opens.

## 8. Feasibility checks

`PROJECT_BRIEF.md` §1.6 asks for cheap aggregate checks at construction, with
network-level routability left to the solver. That division is right, but the
production check as stated there is wrong for this model, and §8.2 of the brief
records the correction.

**What the brief says:** an instance is infeasible if demand in any period
exceeds total plant capacity.

**Why that is too strict:** production can be banked. A peak larger than one
period of output is perfectly servable by building ahead — which is exactly what
the `spike` sample network does, and the brief's rule would have rejected it.

**What actually holds**, both checked in `data/schema.py`:

1. *Cumulative production.* Output moves forward in time but is never borrowed
   from the future, so demand through period $t$ must fit within opening stock
   plus $t$ periods of production:
   $\sum_{s \le t} D_s \le \sum_w I_{w,0} + t \sum_p \overline{Q}_p$.
2. *Per-period outbound throughput.* Whatever is consumed in a period must leave
   a warehouse in that period, so this limit really is per-period and no amount
   of pre-building relaxes it: $\sum_c D_{c,t} \le \sum_w \overline{U}_w$.

Both are necessary but not sufficient. Neither establishes that flow can route
through the network given which facilities are candidates, which is more
expensive to determine and is left to the solver — `solve_scn` names it among
the likely causes when reporting infeasibility.

## 9. A behavior that surprises people

**Opening inventory forces a warehouse open.** There is no disposal variable, so
inventory cannot be written off: $I_{w,1} \ge I_{w,0}$ minus whatever ships out,
and outflow is gated by $y_w$. A warehouse holding opening stock therefore
cannot be closed at any fixed cost. Setting `initial_inventory` quietly
pre-decides that facility. It is tested, and worth knowing before it surprises
someone.

## 10. Out of scope

From `PROJECT_BRIEF.md` §5:

- Multi-product / multi-commodity flows
- Time-varying facility status (phased opening or closing)
- Backorders, stockouts, or lost sales
- Statistically-derived safety stock
- Transportation lead times
- Capacity expansion over time
- Economies of scale or quantity discounts
