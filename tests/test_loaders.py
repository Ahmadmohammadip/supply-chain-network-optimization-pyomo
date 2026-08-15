"""Phase 3: JSON instance loading and CSV demand tables, including the failure
messages."""

import json

import pytest

from scn_opt.data.loaders import (
    load_demand_csv,
    load_demand_csv_text,
    load_system_json,
    system_from_dict,
    system_to_dict,
)

INSTANCE = {
    "plants": [
        {"name": "P1", "capacity": 100.0, "production_cost": 1.0, "fixed_cost": 500.0},
        {"name": "P2", "capacity": 60.0, "production_cost": 0.8, "fixed_cost": 700.0},
    ],
    "warehouses": [
        {
            "name": "W1",
            "throughput_capacity": 80.0,
            "storage_capacity": 40.0,
            "fixed_cost": 200.0,
            "holding_cost": 0.5,
            "safety_stock": 10.0,
        },
        {"name": "W2", "throughput_capacity": 90.0},
    ],
    "customers": [
        {"name": "C1", "demand": [10.0, 20.0, 30.0]},
        {"name": "C2", "demand": [5.0, 5.0, 5.0]},
    ],
    "cost_plant_to_warehouse": [
        {"plant": "P1", "warehouse": "W1", "cost": 2.0},
        {"plant": "P1", "warehouse": "W2", "cost": 3.0},
        {"plant": "P2", "warehouse": "W1", "cost": 4.0},
        {"plant": "P2", "warehouse": "W2", "cost": 1.0},
    ],
    "cost_warehouse_to_customer": [
        {"warehouse": "W1", "customer": "C1", "cost": 3.0},
        {"warehouse": "W1", "customer": "C2", "cost": 6.0},
        {"warehouse": "W2", "customer": "C1", "cost": 7.0},
        {"warehouse": "W2", "customer": "C2", "cost": 2.0},
    ],
}

DEMAND_CSV = (
    "customer,period,demand\n"
    "C1,1,10\nC1,2,20\nC1,3,30\n"
    "C2,1,5\nC2,2,5\nC2,3,5\n"
)


def _write(tmp_path, data, name="instance.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_loads_a_complete_instance(tmp_path):
    system = load_system_json(_write(tmp_path, INSTANCE))

    assert len(system.plants) == 2
    assert len(system.warehouses) == 2
    assert system.n_periods == 3
    assert system.total_demand == pytest.approx(75.0)
    assert system.cost_warehouse_to_customer[("W2", "C2")] == pytest.approx(2.0)


def test_warehouse_storage_defaults_when_absent_from_json(tmp_path):
    system = load_system_json(_write(tmp_path, INSTANCE))
    w2 = next(w for w in system.warehouses if w.name == "W2")

    assert w2.storage_capacity == pytest.approx(90.0)  # falls back to throughput


@pytest.mark.parametrize(
    "key",
    [
        "plants",
        "warehouses",
        "customers",
        "cost_plant_to_warehouse",
        "cost_warehouse_to_customer",
    ],
)
def test_missing_top_level_key_is_rejected(tmp_path, key):
    data = json.loads(json.dumps(INSTANCE))
    del data[key]

    with pytest.raises(ValueError, match=f"missing required key '{key}'"):
        load_system_json(_write(tmp_path, data))


def test_arc_record_missing_a_field_is_rejected(tmp_path):
    data = json.loads(json.dumps(INSTANCE))
    del data["cost_plant_to_warehouse"][1]["cost"]

    with pytest.raises(ValueError, match=r"cost_plant_to_warehouse\[1\] is missing 'cost'"):
        load_system_json(_write(tmp_path, data))


def test_duplicate_arc_record_is_rejected(tmp_path):
    data = json.loads(json.dumps(INSTANCE))
    data["cost_plant_to_warehouse"].append({"plant": "P1", "warehouse": "W1", "cost": 9.0})

    with pytest.raises(ValueError, match="two entries for"):
        load_system_json(_write(tmp_path, data))


def test_invalid_instance_fails_at_load(tmp_path):
    # Validation lives in the schema, so a bad file fails on load rather than
    # after a solve.
    data = json.loads(json.dumps(INSTANCE))
    data["customers"][0]["demand"] = [10.0, 500.0, 30.0]

    with pytest.raises(ValueError, match="demand through period 2 totals 520.0"):
        load_system_json(_write(tmp_path, data))


def test_round_trips_through_a_dict(tmp_path):
    original = load_system_json(_write(tmp_path, INSTANCE))

    restored = system_from_dict(system_to_dict(original))

    assert restored.total_demand == pytest.approx(original.total_demand)
    assert restored.cost_plant_to_warehouse == original.cost_plant_to_warehouse
    assert restored.cost_warehouse_to_customer == original.cost_warehouse_to_customer
    assert [w.storage_capacity for w in restored.warehouses] == [
        w.storage_capacity for w in original.warehouses
    ]


def test_loads_a_demand_csv(tmp_path):
    path = tmp_path / "demand.csv"
    path.write_text(DEMAND_CSV, encoding="utf-8")

    customers = load_demand_csv(path)

    assert {c.name for c in customers} == {"C1", "C2"}
    by_name = {c.name: c.demand for c in customers}
    assert by_name["C1"] == pytest.approx([10.0, 20.0, 30.0])
    assert by_name["C2"] == pytest.approx([5.0, 5.0, 5.0])


def test_demand_csv_text_matches_the_file_loader(tmp_path):
    path = tmp_path / "demand.csv"
    path.write_text(DEMAND_CSV, encoding="utf-8")

    from_file = load_demand_csv(path)
    from_text = load_demand_csv_text(DEMAND_CSV)

    assert [(c.name, c.demand) for c in from_file] == [
        (c.name, c.demand) for c in from_text
    ]


def test_demand_csv_tolerates_a_byte_order_mark():
    customers = load_demand_csv_text("﻿" + DEMAND_CSV)

    assert len(customers) == 2


def test_demand_csv_missing_columns_are_named(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("customer,period\nC1,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"missing required column\(s\) \['demand'\]"):
        load_demand_csv(path)


def test_demand_csv_non_numeric_value_names_the_line(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("customer,period,demand\nC1,1,10\nC1,2,lots\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 3: column 'demand' is not a number"):
        load_demand_csv(path)


def test_demand_csv_rejects_a_duplicate_period(tmp_path):
    path = tmp_path / "dupe.csv"
    path.write_text("customer,period,demand\nC1,1,10\nC1,1,20\n", encoding="utf-8")

    with pytest.raises(ValueError, match="two rows for period 1"):
        load_demand_csv(path)


def test_demand_csv_rejects_customers_covering_different_periods(tmp_path):
    # A missing row is easy to introduce and would otherwise produce customers
    # with mismatched horizons, which the schema would then reject with a less
    # helpful message.
    path = tmp_path / "ragged.csv"
    path.write_text(
        "customer,period,demand\nC1,1,10\nC1,2,20\nC2,1,5\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="must cover the same periods"):
        load_demand_csv(path)


def test_demand_csv_rejects_gappy_periods(tmp_path):
    path = tmp_path / "gappy.csv"
    path.write_text("customer,period,demand\nC1,1,10\nC1,3,30\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"periods must run 1\.\.2 with no gaps"):
        load_demand_csv(path)


def test_demand_csv_header_only_is_rejected(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("customer,period,demand\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no data rows"):
        load_demand_csv(path)
