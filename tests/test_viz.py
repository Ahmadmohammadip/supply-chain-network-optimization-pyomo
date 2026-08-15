"""Phase 5: plotting helpers. These check that each figure builds and carries
the information it is supposed to — not how it looks.

The Agg backend is selected in conftest.py so these run headless.
"""

from pathlib import Path

import pytest

from scn_opt.data.loaders import load_system_json
from scn_opt.model.builder import build_from_system
from scn_opt.solve import solve_scn
from scn_opt.viz import plot_cost_breakdown, plot_inventory_trajectories, plot_network

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_networks"


@pytest.fixture(scope="module")
def baseline():
    system = load_system_json(SAMPLE_DIR / "baseline.json")
    return system, solve_scn(build_from_system(system))


@pytest.fixture(scope="module")
def baseline_open_everything():
    system = load_system_json(SAMPLE_DIR / "baseline.json")
    return system, solve_scn(build_from_system(system, force_open_all=True))


@pytest.fixture(scope="module")
def spike():
    system = load_system_json(SAMPLE_DIR / "spike.json")
    return system, solve_scn(build_from_system(system))


@pytest.mark.parametrize(
    "plot_fn", [plot_network, plot_inventory_trajectories, plot_cost_breakdown]
)
def test_plots_build(baseline, plot_fn):
    system, result = baseline
    fig = plot_fn(system, result)
    assert fig.axes


def test_network_diagram_labels_closed_facilities(baseline):
    # The facilities that were available and rejected are as much a part of the
    # answer as the ones that opened, so they have to appear on the diagram.
    system, result = baseline
    ax = plot_network(system, result).axes[0]

    labels = [t.get_text() for t in ax.texts]
    closed = {p.name for p in system.plants} - set(result.open_plants)
    closed |= {w.name for w in system.warehouses} - set(result.open_warehouses)

    assert closed  # the baseline network is supposed to reject some
    for name in closed:
        assert f"{name}\n(closed)" in labels
    for name in result.open_warehouses:
        assert name in labels


def test_network_diagram_states_the_open_count_in_its_title(baseline):
    system, result = baseline
    title = plot_network(system, result).axes[0].get_title()

    assert f"{len(result.open_plants)}/{len(system.plants)} plants" in title
    assert f"{len(result.open_warehouses)}/{len(system.warehouses)} warehouses" in title


def test_network_diagram_draws_one_line_per_used_lane(baseline):
    system, result = baseline
    ax = plot_network(system, result).axes[0]

    lanes = {(p, w) for (p, w, _) in result.flows_plant_to_warehouse}
    lanes |= {(w, c) for (w, c, _) in result.flows_warehouse_to_customer}

    assert len(ax.get_lines()) == len(lanes)


def test_inventory_chart_plots_only_open_warehouses(baseline):
    system, result = baseline
    ax = plot_inventory_trajectories(system, result).axes[0]

    labelled = {line.get_label() for line in ax.get_lines()}
    for name in result.open_warehouses:
        assert name in labelled
    closed = {w.name for w in system.warehouses} - set(result.open_warehouses)
    for name in closed:
        assert name not in labelled


def test_inventory_chart_starts_from_the_opening_stock(spike):
    system, result = spike
    ax = plot_inventory_trajectories(system, result).axes[0]

    line = next(line for line in ax.get_lines() if line.get_label() == "W1")
    xs, ys = line.get_xdata(), line.get_ydata()

    assert xs[0] == 0  # period 0 is the opening position, not a decision
    assert ys[0] == pytest.approx(system.warehouses[0].initial_inventory)
    # The spike network builds 50 ahead of the period-2 peak.
    assert max(ys) == pytest.approx(50.0, abs=1e-6)


def test_cost_chart_has_one_bar_per_component(baseline):
    system, result = baseline
    ax = plot_cost_breakdown(system, result).axes[0]

    assert len(ax.containers[0]) == len(result.cost_breakdown())


def test_cost_chart_compares_against_the_open_everything_baseline(
    baseline, baseline_open_everything
):
    system, result = baseline
    _, everything = baseline_open_everything

    ax = plot_cost_breakdown(system, result, baseline=everything).axes[0]

    assert len(ax.containers) == 2  # two bar groups
    saving = everything.total_cost - result.total_cost
    assert f"{saving:,.0f}" in ax.get_title()


def test_cost_chart_title_reports_the_plain_total_without_a_baseline(baseline):
    system, result = baseline
    title = plot_cost_breakdown(system, result).axes[0].get_title()

    assert f"{result.total_cost:,.0f}" in title
    assert "saves" not in title
