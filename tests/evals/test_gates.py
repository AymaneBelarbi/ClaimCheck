from pathlib import Path

import pytest

from evals.gates import Gate, GateTable, check_gates, load_gates


def test_load_gates_parses_release_criteria() -> None:
    gates = load_gates(Path("RELEASE_CRITERIA.md"))

    assert gates == {
        "classification.accuracy": Gate(0.95, "min"),
        "extraction.f1": Gate(0.90, "min"),
        "retrieval.recall_at_5": Gate(0.85, "min"),
        "retrieval.wrong_version_hits": Gate(0.0, "max"),
        "validation.accuracy": Gate(1.00, "min"),
        "agent.faithfulness": Gate(0.90, "min"),
    }


def test_load_gates_ignores_prose_and_other_tables(tmp_path: Path) -> None:
    doc = tmp_path / "criteria.md"
    doc.write_text(
        "# Criteria\n\nSome text with a | pipe.\n\n"
        "| metric | gate | direction |\n| --- | --- | --- |\n"
        "| a.b | 0.5 | min |\n| c.d | 1 | max |\n\n"
        "| name | owner |\n| --- | --- |\n| x | y |\n"
    )

    assert load_gates(doc) == {"a.b": Gate(0.5, "min"), "c.d": Gate(1.0, "max")}


def test_a_row_without_direction_is_a_minimum(tmp_path: Path) -> None:
    doc = tmp_path / "criteria.md"
    doc.write_text("| metric | gate |\n| --- | --- |\n| a.b | 0.5 |\n")

    assert load_gates(doc) == {"a.b": Gate(0.5, "min")}


def test_load_gates_without_table_is_an_error(tmp_path: Path) -> None:
    doc = tmp_path / "criteria.md"
    doc.write_text("# nothing here\n")

    with pytest.raises(ValueError, match="gate table"):
        load_gates(doc)


@pytest.mark.parametrize(
    "row",
    [
        "| a.b | 0.5 | Max |",
        "| a.b | 0.5 | sideways |",
        "| a.b | high | min |",
        "| a.b | 0.5 | min | extra |",
        "| a.b |  | min |",
    ],
)
def test_a_metric_row_that_does_not_parse_is_an_error(tmp_path: Path, row: str) -> None:
    doc = tmp_path / "criteria.md"
    doc.write_text(
        "| metric | gate | direction |\n| --- | --- | --- |\n| c.d | 0.9 | min |\n" + row + "\n"
    )

    with pytest.raises(ValueError, match=r"a\.b"):
        load_gates(doc)


def test_check_gates_compares_only_metrics_present_in_the_run() -> None:
    gates = {"classification.accuracy": Gate(0.95, "min"), "extraction.f1": Gate(0.90, "min")}

    table = check_gates({"classification.accuracy": 0.97}, gates)

    assert table.failures == []
    assert [row.metric for row in table.rows] == ["classification.accuracy"]
    assert table.missing == ["extraction.f1"]


def test_check_gates_reports_failures() -> None:
    gates = {"classification.accuracy": Gate(0.95, "min"), "validation.accuracy": Gate(1.0, "min")}

    table = check_gates({"classification.accuracy": 0.94, "validation.accuracy": 1.0}, gates)

    assert [(r.metric, r.value, r.gate) for r in table.failures] == [
        ("classification.accuracy", 0.94, Gate(0.95, "min"))
    ]
    assert "classification.accuracy" in table.render_failures()
    assert "validation.accuracy" not in table.render_failures()
    assert table.missing == []


def test_gate_exactly_at_threshold_passes() -> None:
    assert check_gates({"a.b": 0.95}, {"a.b": Gate(0.95, "min")}).failures == []
    assert check_gates({"a.b": 0.0}, {"a.b": Gate(0.0, "max")}).failures == []


def test_a_max_gate_fails_above_its_threshold() -> None:
    gates = {"a.b": Gate(0.0, "max"), "c.d": Gate(0.5, "max")}
    table = check_gates({"a.b": 1.0, "c.d": 0.2}, gates)

    assert [r.metric for r in table.failures] == ["a.b"]
    rendered = table.render_failures()
    assert "<= 0.0000" in rendered
    assert "1.0000" in rendered


def test_render_missing_lists_each_metric() -> None:
    table = check_gates({}, {"a.b": Gate(0.5, "min"), "c.d": Gate(0.0, "max")})

    assert table.missing == ["a.b", "c.d"]
    assert table.render_missing().splitlines() == ["a.b", "c.d"]


def test_gate_table_type_is_exported() -> None:
    assert GateTable(rows=[]).failures == []
