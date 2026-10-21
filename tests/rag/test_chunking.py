from decimal import Decimal

from claimcheck.rag.chunking import chunk_clauses

POLICY = """# Northgate Mutual private motor policy AUT-104233, version 2

In force from 2025-01-01, until 2026-01-01.

## 4.1 Collision damage

We cover accidental damage caused by a collision, up to a limit of $15,000 per claim.

## 4.2 Glass breakage

We cover the windscreen and side windows, up to a limit of $1,500 per claim.

## 4.3 Theft of the vehicle

We cover loss by theft, up to a limit of $20,000.50 per claim.
"""


def test_splits_on_headings_with_ref_title_and_text() -> None:
    clauses = chunk_clauses(POLICY)

    assert [c.clause_ref for c in clauses] == ["4.1", "4.2", "4.3"]
    assert [c.title for c in clauses] == [
        "Collision damage",
        "Glass breakage",
        "Theft of the vehicle",
    ]
    assert (
        clauses[0].text
        == "We cover accidental damage caused by a collision, up to a limit of $15,000 per claim."
    )


def test_coverage_type_comes_from_the_title() -> None:
    clauses = chunk_clauses(POLICY)

    assert [c.coverage_type for c in clauses] == ["collision", "glass", "theft"]


def test_limit_amount_strips_thousands_separators() -> None:
    clauses = chunk_clauses(POLICY)

    assert clauses[0].limit_amount == Decimal("15000")
    assert clauses[1].limit_amount == Decimal("1500")
    assert clauses[2].limit_amount == Decimal("20000.50")


def test_clause_without_limit_or_coverage_type() -> None:
    markdown = "## 5.1 Excess\n\nYou pay $500 on each claim.\n"

    (clause,) = chunk_clauses(markdown)

    assert clause.coverage_type is None
    assert clause.limit_amount is None


def test_text_before_the_first_heading_is_not_a_clause() -> None:
    assert chunk_clauses("# Title\n\nIn force from 2025-01-01.\n") == []
