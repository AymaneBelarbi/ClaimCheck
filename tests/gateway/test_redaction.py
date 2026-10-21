"""Redaction tests: nothing here needs a key or a network."""

import json
from pathlib import Path
from typing import Any

import pytest
import respx
from pydantic import BaseModel

from claimcheck.core.settings import Settings
from claimcheck.gateway.anthropic import AnthropicGateway
from claimcheck.gateway.base import ChatTurn, LLMGateway, Message, ToolCall, ToolSpec, Usage
from claimcheck.gateway.factory import get_gateway
from claimcheck.gateway.ollama import OllamaGateway
from claimcheck.gateway.redaction import RedactingGateway, Redactor, scope_gateway
from claimcheck.gateway.replay import ReplayGateway
from claimcheck.gateway.tracing import TracingGateway
from tests.fakes import FakeGateway

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
USAGE = Usage(model="fake", input_tokens=1, output_tokens=1, cost_usd=0.0, latency_ms=1.0)


class Quote(BaseModel):
    value: str
    source_text: str


class Answer(BaseModel):
    summary: str
    quotes: list[Quote]
    tags: list[str] = []


def anthropic_reply(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": [{"type": "text", "text": json.dumps(payload)}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


# --- detection ---------------------------------------------------------------


def test_labelled_name_redacted_everywhere() -> None:
    text = (
        "Claimant: Ana Souza\nPolicy AUT-123456\n"
        "The vehicle was driven by Ana Souza.\nSouza signed."
    )
    redacted, mapping = Redactor().redact(text)
    assert "Souza" not in redacted
    assert "Ana" not in redacted
    assert redacted.count("[PERSON_") >= 3
    assert "Ana Souza" in mapping.values()


def test_name_on_the_line_after_the_label() -> None:
    redacted, _ = Redactor().redact("NAME\nBECERRA DESIDERIO\nDATE OF BIRTH\n18 January 1962")
    assert "BECERRA" not in redacted
    assert "DESIDERIO" not in redacted
    assert "DATE OF BIRTH" in redacted


@pytest.mark.parametrize("label", ["Name", "Full name", "Insured", "Claimant", "Driver", "Officer"])
def test_every_label_is_recognised(label: str) -> None:
    redacted, _ = Redactor().redact(f"{label.upper()}: Marta Lindqvist\nother text")
    assert "Lindqvist" not in redacted
    assert "other text" in redacted


def test_name_capture_stops_at_end_of_name() -> None:
    redacted, mapping = Redactor().redact("Driver: Jean-Luc O'Brien signed the form on Monday")
    assert "Jean-Luc O'Brien" in mapping.values()
    assert "signed the form on Monday" in redacted


def test_accented_names_and_particles() -> None:
    redacted, _ = Redactor().redact("Insured: José Álvarez de la Cruz\nNote: José called.")
    assert "José" not in redacted
    assert "Álvarez" not in redacted
    assert "Cruz" not in redacted


def test_name_capture_is_limited_to_five_tokens() -> None:
    _, mapping = Redactor().redact("Name: Ana Maria Clara Beatriz Lucia Souza")
    assert "Ana Maria Clara Beatriz Lucia" in mapping.values()
    assert "Souza" not in " ".join(mapping.values())


def test_short_tokens_are_not_redacted_on_their_own() -> None:
    redacted, _ = Redactor().redact("Claimant: Li Wu Chen\nLi is in the list of tokens.")
    assert "Chen" not in redacted
    # "Li" and "Wu" are under three letters, so only the full name removes them.
    assert "Li is in" in redacted


def test_lowercase_use_of_a_name_token_is_kept() -> None:
    redacted, _ = Redactor().redact("Name: Mark Bishop\nThe bishop moved. Bishop Road is closed.")
    assert "The bishop moved" in redacted
    assert "Bishop Road" not in redacted


def test_unlabelled_text_is_untouched() -> None:
    text = "Rear bumper cover, replace\nTotal $8,564.37"
    redacted, mapping = Redactor().redact(text)
    assert redacted == text
    assert mapping == {}


def test_id_number_redacted() -> None:
    redacted, mapping = Redactor().redact("DL NO.\nD30736241\nPassport XK1234567 on file")
    assert "D30736241" not in redacted
    assert "XK1234567" not in redacted
    assert sorted(mapping.values()) == ["D30736241", "XK1234567"]
    assert all(key.startswith("[ID_") for key in mapping)


def test_policy_number_kept() -> None:
    text = "Policy number AUT-104233\nDL NO. D30736241\nPlate FCV-6316 invoice INV-96538"
    redacted, _ = Redactor().redact(text)
    assert "AUT-104233" in redacted
    assert "FCV-6316" in redacted
    assert "INV-96538" in redacted
    assert "D30736241" not in redacted


def test_same_value_gets_the_same_placeholder() -> None:
    redactor = Redactor()
    redacted, mapping = redactor.redact("ID D30736241 and again D30736241, other D11111111")
    assert redacted.count("[ID_1]") == 2
    assert "[ID_2]" in redacted
    assert mapping == {"[ID_1]": "D30736241", "[ID_2]": "D11111111"}


def test_placeholders_are_stable_across_texts_of_one_redactor() -> None:
    redactor = Redactor()
    first, _ = redactor.redact("Claimant: Ana Souza")
    second, mapping = redactor.redact("Later: Ana Souza called")
    assert first.split(": ")[1] == second.split(": ")[1].split(" called")[0]
    assert len(mapping) == len(set(mapping.values()))


def test_a_name_learned_in_one_text_is_redacted_in_the_next() -> None:
    redactor = Redactor()
    redactor.redact("Claimant: Ana Souza")
    redacted, _ = redactor.redact("Please summarise the file of Ana Souza.")
    assert "Souza" not in redacted


# --- expanded labels ---------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "name", "kept"),
    [
        ("Invoice INV-1 Customer: Angela Mann\nIssued", "Mann", "Issued"),
        ("Tax invoice\nBill to Julia Cardoso\nVehicle VPM-0050", "Cardoso", "Vehicle VPM-0050"),
        ("Billed to: Roland Leclerc", "Leclerc", ""),
        ("Claimant name Desiderio Becerra\nPolicy number", "Becerra", "Policy number"),
        ("POLICYHOLDER / CLAIMANT\nJulia Cardoso\nTELEPHONE", "Cardoso", "TELEPHONE"),
        ("Policy holder: Marta Lindqvist", "Lindqvist", ""),
        ("Insured name: Marta Lindqvist", "Lindqvist", ""),
        ("Name of driver: Marta Lindqvist", "Lindqvist", ""),
        ("Holder\nMarta Lindqvist", "Lindqvist", ""),
        ("Reporting officer Officer Fletcher (badge 2904)", "Fletcher", "(badge 2904)"),
        ("Report taken by Officer Reyes (badge 4411)", "Reyes", "(badge 4411)"),
        ("Full name Marta Lindqvist", "Lindqvist", ""),
    ],
)
def test_expanded_labels(text: str, name: str, kept: str) -> None:
    redacted, _ = Redactor().redact(text)
    assert name not in redacted
    assert kept in redacted


def test_title_word_is_not_redacted_as_a_name() -> None:
    redacted, mapping = Redactor().redact("Officer Reyes took the report.\nThe officer left.")
    assert "Officer [PERSON_1] took" in redacted
    assert list(mapping.values()) == ["Reyes"]


def test_initials_stay_part_of_the_name() -> None:
    _, mapping = Redactor().redact("Claimant: John A. Smith signed")
    assert "John A. Smith" in mapping.values()


def test_short_labels_do_not_swallow_form_headings() -> None:
    text = "DRIVER LICENCE Dept. of Motor Vehicles\nNAME\nDATE OF BIRTH\n18 January 1962"
    redacted, mapping = Redactor().redact(text)
    assert redacted == text
    assert mapping == {}


def test_accent_insensitive_matching_restores_original_spelling() -> None:
    redactor = Redactor()
    redacted, mapping = redactor.redact("Claimant: Júlia Cardoso\nID card: JULIA CARDOSO")
    assert "ardoso" not in redacted
    assert "ULIA" not in redacted
    assert set(mapping.values()) >= {"Júlia Cardoso", "JULIA CARDOSO"}


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("Claimant: Van Dyke", "Dyke"),
        ("Name: De Silva", "Silva"),
        ("Name: Al Rashid", "Rashid"),
        ("Name: Le Blanc", "Blanc"),
        ("Name: van der Berg", "Berg"),
    ],
)
def test_names_that_start_with_a_particle(text: str, name: str) -> None:
    redacted, mapping = Redactor().redact(f"{text}\nlater: {text.split(': ')[1]} signed")
    assert name not in redacted
    assert text.split(": ")[1] in mapping.values()


def test_a_lone_particle_is_not_a_name() -> None:
    redacted, mapping = Redactor().redact("Name: Van\nVan hire")
    assert redacted == "Name: Van\nVan hire"
    assert mapping == {}


def test_surname_comma_firstname() -> None:
    redacted, mapping = Redactor().redact("Name: Smith, John\nJohn called, Smith answered")
    assert "John" not in redacted
    assert "Smith" not in redacted
    assert "Smith, John" in mapping.values()


def test_trailing_comma_is_not_part_of_the_name() -> None:
    _, mapping = Redactor().redact("Claimant: Ana Souza, signed on Monday")
    assert list(mapping.values()) == ["Ana Souza"]


# --- packet scope ------------------------------------------------------------

CLAIM_FORM = "Claimant name\nDesiderio Becerra\nPolicy number AUT-104233"
LICENCE = "DRIVER LICENCE\nBECERRA DESIDERIO\nLicence no. D30736241"


def test_name_labelled_in_one_document_is_redacted_in_another() -> None:
    inner = FakeGateway(
        structured_handler=lambda call: Answer(summary="x", quotes=[]), is_cloud=True
    )
    gateway = RedactingGateway(inner).scoped([CLAIM_FORM, LICENCE])
    gateway.structured(
        system="s",
        prompt="DRIVER LICENCE\nBECERRA\nSigned Desiderio Becerra",
        schema=Answer,
        purpose="t",
    )
    sent = inner.calls[0].prompt
    assert sent is not None
    assert "ECERRA" not in sent
    assert "esiderio" not in sent
    assert "DRIVER LICENCE" in sent


def test_an_all_caps_bare_name_line_is_covered_by_the_scope() -> None:
    inner = FakeGateway(
        structured_handler=lambda call: Answer(summary="x", quotes=[]), is_cloud=True
    )
    claim = "Claimant name Angela Mann\nPolicy number AUT-104233"
    gateway = RedactingGateway(inner).scoped([claim])
    gateway.structured(
        system="s",
        prompt="DRIVER LICENCE\nANGELA MANN\nLicence no. D52474179",
        schema=Answer,
        purpose="t",
    )
    sent = inner.calls[0].prompt
    assert sent is not None
    assert "ANGELA" not in sent
    assert "MANN" not in sent


def test_scope_keeps_placeholders_per_call() -> None:
    inner = FakeGateway(
        structured_handler=lambda call: Answer(summary="[PERSON_1]", quotes=[]), is_cloud=True
    )
    gateway = RedactingGateway(inner).scoped([CLAIM_FORM])
    first = gateway.structured(system="s", prompt="Ask Becerra", schema=Answer, purpose="t")
    second = gateway.structured(system="s", prompt="Ask Becerra", schema=Answer, purpose="t")
    assert first.output.summary == "Becerra"
    assert second.output.summary == "Becerra"
    assert inner.calls[0].prompt == inner.calls[1].prompt == "Ask [PERSON_1]"


def test_scoping_does_not_change_the_original_gateway() -> None:
    inner = FakeGateway(
        structured_handler=lambda call: Answer(summary="x", quotes=[]), is_cloud=True
    )
    plain = RedactingGateway(inner)
    plain.scoped([CLAIM_FORM])
    plain.structured(system="s", prompt="Ask Becerra", schema=Answer, purpose="t")
    assert inner.calls[0].prompt == "Ask Becerra"


def test_scope_gateway_passes_a_local_gateway_through() -> None:
    local = FakeGateway(is_cloud=False)
    assert scope_gateway(local, [CLAIM_FORM]) is local


def test_scope_gateway_scopes_a_redacting_gateway_over_replay(tmp_path: Path) -> None:
    inner = FakeGateway(
        structured_handler=lambda call: Answer(summary="x", quotes=[]), is_cloud=True
    )
    replay = ReplayGateway(inner, tmp_path, record=True)
    gateway: LLMGateway = RedactingGateway(replay)
    scoped = scope_gateway(gateway, [CLAIM_FORM])
    assert isinstance(scoped, RedactingGateway)
    assert scoped is not gateway
    assert scoped.inner is replay
    scoped.structured(system="s", prompt="Ask Becerra", schema=Answer, purpose="t", cache_key="k")
    assert "Becerra" not in "".join(p.read_text() for p in tmp_path.glob("*.json"))


def test_scope_gateway_from_the_factory_leaves_ollama_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = get_gateway(make_settings(monkeypatch, llm_backend="ollama"))
    scoped = scope_gateway(gateway, [CLAIM_FORM])
    # Only the tracing wrapper is rebuilt; nothing redacting is added.
    assert isinstance(scoped, TracingGateway)
    assert scoped.inner is gateway.inner  # type: ignore[attr-defined]


# --- restore -----------------------------------------------------------------


def test_restore_roundtrip() -> None:
    redactor = Redactor()
    _, mapping = redactor.redact("Claimant: Ana Souza\nDL D30736241")
    placeholder = next(k for k, v in mapping.items() if v == "Ana Souza")
    id_placeholder = next(k for k, v in mapping.items() if v == "D30736241")
    answer = Answer(
        summary=f"{placeholder} holds licence {id_placeholder}.",
        quotes=[Quote(value=placeholder, source_text=f"Claimant: {placeholder}")],
        tags=[f"owner={placeholder}", "plain"],
    )
    restored = redactor.restore(answer, mapping)
    assert restored.summary == "Ana Souza holds licence D30736241."
    assert restored.quotes[0].value == "Ana Souza"
    assert restored.quotes[0].source_text == "Claimant: Ana Souza"
    assert restored.tags == ["owner=Ana Souza", "plain"]
    # The input is left alone.
    assert placeholder in answer.summary


def test_restore_keeps_unknown_placeholders() -> None:
    redactor = Redactor()
    _, mapping = redactor.redact("Claimant: Ana Souza")
    answer = Answer(summary="[PERSON_99] and [ID_7]", quotes=[])
    assert redactor.restore(answer, mapping).summary == "[PERSON_99] and [ID_7]"


def test_restore_returns_exact_surface_form() -> None:
    # Quotes are matched against the OCR text later, so the case must come back as written.
    redactor = Redactor()
    redacted, mapping = redactor.redact("Claimant: Desiderio Becerra\nSIGNED BY BECERRA")
    answer = Answer(summary=redacted, quotes=[])
    assert (
        redactor.restore(answer, mapping).summary
        == "Claimant: Desiderio Becerra\nSIGNED BY BECERRA"
    )


# --- gateway -----------------------------------------------------------------

CLAIM = "Claimant: Ana Souza\nLicence D30736241\nPolicy AUT-123456\nAna Souza signed."


def redacting(inner: FakeGateway) -> RedactingGateway:
    return RedactingGateway(inner, Redactor())


def test_gateway_exposes_inner_identity() -> None:
    inner = FakeGateway(model="m-1", is_cloud=True)
    gateway = redacting(inner)
    assert gateway.is_cloud is True
    assert gateway.model == "m-1"


def test_structured_redacts_prompt_and_system_and_restores_output() -> None:
    def handler(call: Any) -> BaseModel:
        placeholder = call.prompt.split("Claimant: ")[1].split("\n")[0]
        return Answer(
            summary=f"{placeholder} signed",
            quotes=[Quote(value=placeholder, source_text=f"{placeholder} signed.")],
        )

    inner = FakeGateway(structured_handler=handler, is_cloud=True)
    result = redacting(inner).structured(
        system="Review the file of Ana Souza only.",
        prompt=CLAIM,
        schema=Answer,
        purpose="test",
        cache_key="k1",
    )
    sent = inner.calls[0]
    assert sent.prompt is not None
    assert "Souza" not in sent.prompt
    assert "D30736241" not in sent.prompt
    assert "AUT-123456" in sent.prompt
    assert "Souza" not in sent.system
    assert sent.cache_key == "k1"
    assert sent.purpose == "test"
    assert result.output.summary == "Ana Souza signed"
    assert result.output.quotes[0].source_text == "Ana Souza signed."
    assert result.usage.model == "fake-model"


def test_chat_redacts_every_message_and_tool_call_and_restores_reply() -> None:
    turn = ChatTurn(
        text="[PERSON_1] is covered",
        tool_calls=[ToolCall(id="c2", name="lookup", arguments={"who": "[PERSON_1]", "n": 3})],
        usage=USAGE,
    )
    inner = FakeGateway(chat_turns=[turn], is_cloud=True)
    messages = [
        Message(role="user", content=CLAIM),
        Message(
            role="assistant",
            content="Looking up Ana Souza",
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="lookup",
                    arguments={"who": "Ana Souza", "nested": {"ids": ["D30736241"]}},
                )
            ],
        ),
        Message(role="tool", content="Record for Ana Souza, D30736241", tool_call_id="c1"),
    ]
    out = redacting(inner).chat(
        system="Ask about Ana Souza", messages=messages, tools=[], cache_key=None
    )
    wire = json.dumps(
        [inner.calls[0].system, [m.model_dump(mode="json") for m in inner.calls[0].messages]]
    )
    assert "Souza" not in wire
    assert "D30736241" not in wire
    assert "AUT-123456" in wire
    sent_call = inner.calls[0].messages[1].tool_calls[0]
    assert sent_call.id == "c1"
    assert sent_call.arguments["who"].startswith("[PERSON_")
    assert out.text == "Ana Souza is covered"
    assert out.tool_calls[0].arguments == {"who": "Ana Souza", "n": 3}
    assert out.tool_calls[0].id == "c2"
    # The caller's messages are not modified.
    assert messages[0].content == CLAIM


def test_each_call_gets_its_own_mapping() -> None:
    inner = FakeGateway(
        structured_handler=lambda call: Answer(summary="[PERSON_1]", quotes=[]), is_cloud=True
    )
    gateway = redacting(inner)
    first = gateway.structured(system="s", prompt="Name: Ana Souza", schema=Answer, purpose="t")
    second = gateway.structured(system="s", prompt="Name: Pia Larsen", schema=Answer, purpose="t")
    assert first.output.summary == "Ana Souza"
    assert second.output.summary == "Pia Larsen"


def test_cloud_backend_receives_no_raw_name() -> None:
    with respx.mock:
        route = respx.post(ANTHROPIC_URL).respond(
            json=anthropic_reply(
                {
                    "summary": "[PERSON_1] holds [ID_1]",
                    "quotes": [{"value": "[PERSON_1]", "source_text": "Claimant: [PERSON_1]"}],
                    "tags": [],
                }
            )
        )
        gateway = RedactingGateway(
            AnthropicGateway(api_key="sk-test", model="claude-haiku-5-5"), Redactor()
        )
        result = gateway.structured(
            system="You read claim forms.", prompt=CLAIM, schema=Answer, purpose="test"
        )
        body = route.calls.last.request.content.decode()
    assert "Souza" not in body
    assert "Ana" not in body
    assert "D30736241" not in body
    assert "AUT-123456" in body
    assert result.output.summary == "Ana Souza holds D30736241"
    assert result.output.quotes[0].source_text == "Claimant: Ana Souza"


def test_cloud_chat_receives_no_raw_name() -> None:
    reply = {
        "id": "m",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": [{"type": "text", "text": "Done for [PERSON_1]"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    with respx.mock:
        route = respx.post(ANTHROPIC_URL).respond(json=reply)
        gateway = RedactingGateway(
            AnthropicGateway(api_key="sk-test", model="claude-haiku-5-5"), Redactor()
        )
        turn = gateway.chat(
            system="s",
            messages=[Message(role="user", content=CLAIM)],
            tools=[ToolSpec(name="t", description="d", input_schema={"type": "object"})],
        )
        body = route.calls.last.request.content.decode()
    assert "Souza" not in body
    assert turn.text == "Done for Ana Souza"


def test_redacting_gateway_passes_failures_through() -> None:
    boom = RuntimeError("provider down")
    inner = FakeGateway(structured_outputs=[boom], is_cloud=True)
    with pytest.raises(RuntimeError, match="provider down"):
        redacting(inner).structured(system="s", prompt=CLAIM, schema=Answer, purpose="t")


def test_replay_counters_stay_reachable_and_cassettes_hold_redacted_text(
    tmp_path: Path,
) -> None:
    inner = FakeGateway(
        structured_handler=lambda call: Answer(summary="x", quotes=[]), is_cloud=True
    )
    replay = ReplayGateway(inner, tmp_path, record=True)
    gateway = RedactingGateway(replay, Redactor())
    gateway.structured(system="s", prompt=CLAIM, schema=Answer, purpose="t", cache_key="k")
    assert gateway.inner is replay
    assert gateway.drift_count == 0
    assert gateway.drifted_keys == []
    # The cassette holds the redacted request only.
    stored = "".join(p.read_text() for p in tmp_path.glob("*.json"))
    assert "Souza" not in stored


# --- factory -----------------------------------------------------------------


def make_settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    for name, value in env.items():
        monkeypatch.setenv(name.upper(), value)
    return Settings(_env_file=None)


def _inside_tracing(gateway: LLMGateway) -> LLMGateway:
    """What get_gateway wraps in tracing, which is always outermost."""
    assert isinstance(gateway, TracingGateway)
    return gateway.inner


def test_ollama_not_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    gateway = _inside_tracing(get_gateway(make_settings(monkeypatch, llm_backend="ollama")))
    assert isinstance(gateway, OllamaGateway)


@pytest.mark.parametrize(
    ("backend", "key_var"), [("anthropic", "anthropic_api_key"), ("openai", "openai_api_key")]
)
def test_cloud_backends_are_wrapped(
    monkeypatch: pytest.MonkeyPatch, backend: str, key_var: str
) -> None:
    gateway = _inside_tracing(
        get_gateway(make_settings(monkeypatch, llm_backend=backend, **{key_var: "k"}))
    )
    assert isinstance(gateway, RedactingGateway)
    assert gateway.is_cloud is True


def test_replay_over_a_cloud_backend_is_wrapped_and_recorded_redacted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gateway = _inside_tracing(
        get_gateway(
            make_settings(
                monkeypatch,
                llm_backend="replay",
                replay_dir=str(tmp_path),
                replay_record="true",
                replay_inner="anthropic",
                anthropic_api_key="k",
            )
        )
    )
    assert isinstance(gateway, RedactingGateway)
    assert isinstance(gateway.inner, ReplayGateway)
    assert gateway.is_cloud is True


def test_replay_over_ollama_is_not_wrapped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    gateway = _inside_tracing(
        get_gateway(
            make_settings(
                monkeypatch,
                llm_backend="replay",
                replay_dir=str(tmp_path),
                replay_record="true",
                replay_inner="ollama",
            )
        )
    )
    assert isinstance(gateway, ReplayGateway)


def test_replaying_cassettes_recorded_from_a_cloud_backend_redacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "k.json").write_text(
        '{"key": "k", "request_sha256": "x", "kind": "chat", "model": "m",'
        ' "is_cloud": true, "response": {}}'
    )
    gateway = _inside_tracing(
        get_gateway(make_settings(monkeypatch, llm_backend="replay", replay_dir=str(tmp_path)))
    )
    assert isinstance(gateway, RedactingGateway)
