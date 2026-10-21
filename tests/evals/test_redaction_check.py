from datetime import date

from core.types import ClaimType, DocType, Verdict
from evals.dataset.types import GroundTruth, GTDocument, Scenario
from evals.redaction_check import name_tokens, occurrences, people, redactor


def _doc(doc_type: DocType, **fields: str) -> GTDocument:
    return GTDocument(
        filename=f"{doc_type.value}.pdf",
        doc_type=doc_type,
        fields=dict(fields),
        field_verdicts=dict.fromkeys(fields, Verdict.FOUND),
    )


def test_tokens_skip_particles_and_ignore_accents() -> None:
    assert name_tokens("Lourenço de Sá") == {"lourenco", "sa"}


def test_only_capitalized_occurrences_count() -> None:
    assert occurrences("Mark said mark the MARK", {"mark"}) == ["Mark", "MARK"]


def test_people_are_the_claimant_and_the_licence_holder_once_each() -> None:
    truth = GroundTruth(
        packet_id="p1",
        policy_number="AUT-000001",
        holder_id="cust-1",
        claim_type=ClaimType.GLASS,
        incident_date=date(2026, 1, 2),
        scenario=Scenario.CLEAN,
        variant=None,
        documents=[
            _doc(DocType.CLAIM_FORM, claimant_name="Ana Díaz"),
            _doc(DocType.ID_CARD, full_name="DIAZ ANA", id_number="D12345678"),
            _doc(DocType.INVOICE, vendor="Diaz Auto Glass"),
        ],
        rule_verdicts={},
    )

    assert people(truth) == ["Ana Díaz"]


def test_the_configurations_differ_only_in_scope_and_labels() -> None:
    packet = ["Bill to Ana Souza", "SOUZA\nANA"]
    licence = packet[1]

    production, _ = redactor("production", packet).fresh().redact(licence)
    unscoped, _ = redactor("no_packet_scope", packet).fresh().redact(licence)
    six, _ = redactor("first_six_labels", packet).fresh().redact(licence)

    assert "SOUZA" not in production
    assert "SOUZA" in unscoped
    # "Bill to" followed by spaces is not one of the first six labels.
    assert "SOUZA" in six
