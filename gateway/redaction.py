"""Redact person names and ID numbers before text reaches a cloud LLM.

`get_gateway` wraps every backend whose `is_cloud` is true in a `RedactingGateway`; the local
Ollama backend is never wrapped and receives the raw text. Each call builds its own
`Redactor`, so placeholders mean the same thing within one request and nothing carries over
between requests. Placeholders in the model's answer are swapped back for the original text,
including inside quotes (`source_text` is later matched against the raw OCR text, so every
string in the output is restored exactly as it was written). A placeholder that is not in the
mapping is left as the model wrote it; the redactor never invents a value.

What is detected
- Names: the value after a label (Name, Full name, Insured, Claimant, Driver, Customer,
  Policyholder, Holder, Bill to, Officer and a few longer forms such as "Claimant name" or
  "Reporting officer"; any case). The value follows a colon, starts on the next line, or, for
  the specific labels in SPACE_LABELS, follows plain spaces ("Bill to Julia Cardoso"). The
  short generic labels do not take a space separator, so "DRIVER LICENCE" is not a name. A
  value runs to the end of the line, at most five tokens. A name token starts with a capital
  letter, an initial ("A."), or is a lowercase particle such as "de" or "van"; the first other
  word (lowercase, digit, bracket, or a form word such as "Date" or "Number") ends the name.
  Leading titles such as "Officer" or "Dr" are skipped. Every occurrence of a captured name is
  then replaced, and so is every capitalized occurrence of each of its tokens of three or more
  letters, ignoring accents ("JULIA" matches a captured "Júlia"). Each distinct spelling gets
  its own placeholder, so "Becerra" and "BECERRA" restore to exactly what was sent.
- Packet scope: the documents of one claim packet share people, and a name that is labelled on
  the claim form is a bare line on the licence. `RedactingGateway.scoped(texts)` (or
  `scope_gateway(gateway, texts)`, which is a no-op for a local backend) returns a gateway whose
  calls already know every name labelled anywhere in `texts`. Placeholders are still numbered
  per call.
- ID numbers: one letter plus eight digits (the licence format of the synthetic dataset) and
  the generic one to three capitals plus six to nine digits. Policy numbers such as
  AUT-123456 have a hyphen and do not match; they are not personal data and the model needs
  them.

Measured by `python -m evals.redaction_check --split all` on the 50 generated packets (170
documents, OCR text, a fresh scope per packet; gold names and ID numbers from ground truth;
report `evals/reports/2026-10-21-redaction.json`): 0 of 43 licence numbers leak. 16 of 143
document/name pairs leak (31 of 372 capitalized name-token occurrences): claim forms 5 of 53,
licences 9 of 50, invoices 2 of 40. Without packet scope the same labels leave 44 of 143 pairs;
with packet scope but only the labels Name, Full name, Insured, Claimant, Driver and Officer
(colon or next-line values only), 52 of 143. The residual comes from the limitations below.

Known limitations (ADR-006 repeats these)
- A name that never follows one of the labels is not found. A claimant described only in
  free text ("my neighbour John Reilly hit me") goes out as written, unless the same name was
  also captured from a label elsewhere in the request or packet scope.
- Forms rendered as a table come out of OCR as a column of labels followed by a column of
  values ("Claimant name", "Policy number", ..., then "Abdul Woods"); the name is then not next
  to its label and is only caught when another document of the packet labels it.
- A bare name line on a licence is only caught through packet scope. When the licence holder
  is not the claimant (a name mismatch, the case a reviewer most needs to see) the holder's
  first name goes out unless the same person is labelled in another document.
- OCR errors defeat matching: "Iain" read as "lain", or "Lourenço" read as "Lourencgo", is a
  different string. A name token shorter than three letters is only replaced as part of the
  full name.
- A space-separated label takes the capitalized word after it even when it is not a name
  ("Customer Receipt" hides "Receipt"), and "Surname, Firstname" takes a capitalized word after
  the comma ("Smith, Lisbon"). That costs context in the prompt only; outputs are restored.
- Names that OCR returned in lowercase are not captured, since a name has to start with a
  capital letter (or be a particle).
- A label with no value, followed by an unrelated capitalized line, captures that line as if
  it were a name, unless the line starts with one of the form words in the stop list.
- Names split across OCR lines are only caught through their individual tokens.
- Common words used as names ("Will", "Mark") are redacted wherever they appear capitalized,
  which costs context but not safety.
- Addresses, phone numbers, plates and dates of birth are not redacted. Redaction covers
  names and ID numbers only.
"""

import re
import unicodedata
from collections.abc import Callable, Iterable
from typing import Any

from pydantic import BaseModel

from gateway.base import ChatTurn, LLMGateway, LLMResult, Message, ToolCall, ToolSpec

# Labels whose value follows a colon or starts on the next line.
NAME_LABELS = (
    "full name",
    "name",
    "insured",
    "insured name",
    "claimant",
    "claimant name",
    "driver",
    "driver name",
    "name of driver",
    "customer",
    "policyholder",
    "policy holder",
    "holder",
    "bill to",
    "billed to",
    "officer",
    "reporting officer",
)
# Labels specific enough to also be followed by the name after plain spaces ("Bill to Julia
# Cardoso"). The short generic ones are left out: "Driver Licence" must not read as a name.
SPACE_LABELS = (
    "full name",
    "insured name",
    "claimant name",
    "driver name",
    "name of driver",
    "customer",
    "policyholder",
    "policy holder",
    "bill to",
    "billed to",
    "officer",
    "reporting officer",
)

# The licence format of the synthetic dataset, then the generic shape. The dataset's own
# pattern lives in evals and is mirrored here on purpose: the gateway must not import evals.
_ID_PATTERN = re.compile(r"\b[A-Z]\d{8}\b|\b[A-Z]{1,3}\d{6,9}\b")

_MAX_NAME_TOKENS = 5
_MIN_TOKEN_LENGTH = 3
_PARTICLES = frozenset(
    [
        "de",
        "da",
        "do",
        "dos",
        "das",
        "del",
        "della",
        "di",
        "van",
        "von",
        "der",
        "den",
        "la",
        "le",
        "el",
        "al",
        "bin",
        "ibn",
        "ben",
        "st",
    ]
)
# Capitalized words that follow a label on forms without being part of a name.
_FORM_WORDS = frozenset(
    [
        "name",
        "names",
        "number",
        "no",
        "nr",
        "date",
        "dob",
        "birth",
        "of",
        "telephone",
        "phone",
        "tel",
        "mobile",
        "email",
        "address",
        "signature",
        "signed",
        "licence",
        "license",
        "vehicle",
        "policy",
        "claim",
        "incident",
        "type",
        "class",
        "born",
        "valid",
        "issued",
        "expires",
        "expiry",
        "registration",
        "description",
        "total",
        "amount",
        "invoice",
        "item",
        "work",
        "copy",
        "service",
        "services",
        "details",
        "information",
        "report",
        "badge",
        "id",
        "the",
        "and",
        "for",
    ]
)
# Titles are skipped at the start of a name: "Reporting officer Officer Fletcher" is Fletcher.
_TITLES = frozenset(
    ["officer", "mr", "mrs", "ms", "miss", "mx", "dr", "sgt", "sergeant", "constable", "inspector"]
)
_JOINERS = r"['\u2019\-]"
_WORD = rf"[^\W\d_]+(?:{_JOINERS}[^\W\d_]+)*"
_WORD_RE = re.compile(_WORD)
_INITIAL_RE = re.compile(r"[^\W\d_]\.(?=\s|$)")
_PLACEHOLDER_RE = re.compile(r"\[(?:PERSON|ID)_\d+\]")


def _alternatives(labels: tuple[str, ...]) -> str:
    # Longest first, so "claimant name" wins over "claimant".
    ordered = sorted(labels, key=len, reverse=True)
    return "|".join(r"[ \t]+".join(map(re.escape, label.split())) for label in ordered)


def _label_patterns(
    labels: tuple[str, ...], space_labels: tuple[str, ...]
) -> list[re.Pattern[str]]:
    patterns = [
        re.compile(
            rf"(?<!\w)(?:{_alternatives(labels)})[ \t]*"
            r"(?::[ \t]*(?:\r?\n[ \t]*)?|\r?\n[ \t]*)(?P<value>[^\r\n]*)",
            re.IGNORECASE,
        )
    ]
    if space_labels:
        patterns.append(
            re.compile(
                rf"(?<!\w)(?:{_alternatives(space_labels)})[ \t]+(?P<value>[^\r\n]*)",
                re.IGNORECASE,
            )
        )
    return patterns


def _fold(text: str) -> str:
    """Strip accents one character at a time, so offsets in the result match the input."""
    return "".join((unicodedata.normalize("NFD", ch) or ch)[0] for ch in text)


def _name_from(line: str) -> str | None:
    """The leading name in `line`, or None when it does not start with one.

    One comma is allowed after a word so "Smith, John" is a single name; a comma that is not
    followed by a name word is dropped.
    """
    tokens: list[str] = []
    comma_used = False
    position = 0
    while len(tokens) < _MAX_NAME_TOKENS:
        start = _skip_blanks(line, position)
        initial = _INITIAL_RE.match(line, start)
        match = initial or _WORD_RE.match(line, start)
        if match is None:
            break
        word = match.group(0)
        lowered = word.lower()
        if initial is None and not tokens and lowered in _TITLES:
            position = match.end()
            continue
        if initial is None and (lowered in _FORM_WORDS or not _name_word(word)):
            break
        if initial is not None and not word[0].isupper():
            break
        tokens.append(word)
        position = match.end()
        after = _skip_blanks(line, position)
        if not comma_used and line.startswith(",", after):
            comma_used = True
            tokens[-1] += ","
            position = after + 1
    if tokens and tokens[-1].endswith(","):
        tokens[-1] = tokens[-1][:-1]
    while tokens and tokens[-1].lower() in _PARTICLES:
        tokens.pop()
    # "Van Dyke" and "Al Rashid" are names; a particle with nothing after it is not.
    if all(t.lower().rstrip(",") in _PARTICLES or _INITIAL_RE.fullmatch(t) for t in tokens):
        return None
    return " ".join(tokens)


def _name_word(word: str) -> bool:
    return word[0].isupper() or word.lower() in _PARTICLES


def _skip_blanks(line: str, position: int) -> int:
    while position < len(line) and line[position] in " \t":
        position += 1
    return position


def _tokens_of(name: str) -> set[str]:
    """Words of a name that are long enough to redact on their own, with hyphen parts."""
    found: set[str] = set()
    for word in _WORD_RE.findall(name):
        parts = [word, *re.split(_JOINERS, word)]
        for part in parts:
            if len(part) >= _MIN_TOKEN_LENGTH and part.lower() not in _PARTICLES:
                found.add(part)
    return found


class Redactor:
    """Replaces names and ID numbers with placeholders and puts them back.

    One instance holds the mapping for one request. `fresh()` gives an empty instance with the
    same configuration.
    """

    def __init__(
        self, labels: tuple[str, ...] = NAME_LABELS, space_labels: tuple[str, ...] = SPACE_LABELS
    ) -> None:
        self._labels = labels
        self._space_labels = space_labels
        self._label_res = _label_patterns(labels, space_labels)
        self._full_names: dict[str, str] = {}  # casefolded, single-spaced -> as captured
        self._tokens: set[str] = set()
        self._name_re: re.Pattern[str] | None = None
        self._placeholder_of: dict[str, str] = {}
        self._mapping: dict[str, str] = {}
        self._counts = {"PERSON": 0, "ID": 0}

    @property
    def mapping(self) -> dict[str, str]:
        return dict(self._mapping)

    def fresh(self) -> "Redactor":
        """An instance with the same configuration and known names but no placeholders."""
        other = Redactor(self._labels, self._space_labels)
        other._full_names = dict(self._full_names)
        other._tokens = set(self._tokens)
        return other

    def learn(self, text: str) -> None:
        """Remember the names labelled in `text` so later redactions cover them."""
        for pattern in self._label_res:
            for match in pattern.finditer(text):
                name = _name_from(match.group("value"))
                if name is None:
                    continue
                key = _fold(" ".join(name.split())).casefold()
                if key in self._full_names:
                    continue
                self._full_names[key] = name
                self._tokens |= _tokens_of(name)
                self._name_re = None

    def learn_from(self, texts: Iterable[str]) -> None:
        """Learn the labelled names of several texts, e.g. every document of one packet."""
        for text in texts:
            self.learn(text)

    def redact(self, text: str) -> tuple[str, dict[str, str]]:
        """Return the redacted text and the placeholder mapping accumulated so far."""
        self.learn(text)
        text = _ID_PATTERN.sub(lambda m: self._placeholder("ID", m.group(0)), text)
        pattern = self._names_pattern()
        if pattern is not None:
            # Match on the accent-stripped text, replace in the original.
            pieces: list[str] = []
            last = 0
            for match in pattern.finditer(_fold(text)):
                pieces.append(text[last : match.start()])
                pieces.append(self._replace_name(text[match.start() : match.end()]))
                last = match.end()
            pieces.append(text[last:])
            text = "".join(pieces)
        return text, self.mapping

    def restore[T: BaseModel](self, obj: T, mapping: dict[str, str]) -> T:
        """A copy of `obj` with every placeholder in every string swapped back."""
        restored = _map_strings(obj, lambda s: self.restore_text(s, mapping))
        assert isinstance(restored, type(obj))
        return restored

    def restore_text(self, text: str, mapping: dict[str, str]) -> str:
        return _PLACEHOLDER_RE.sub(lambda m: mapping.get(m.group(0), m.group(0)), text)

    def _placeholder(self, kind: str, raw: str) -> str:
        known = self._placeholder_of.get(raw)
        if known is not None:
            return known
        self._counts[kind] += 1
        placeholder = f"[{kind}_{self._counts[kind]}]"
        self._placeholder_of[raw] = placeholder
        self._mapping[placeholder] = raw
        return placeholder

    def _replace_name(self, surface: str) -> str:
        is_full_name = _fold(" ".join(surface.split())).casefold() in self._full_names
        if is_full_name or surface[0].isupper():
            return self._placeholder("PERSON", surface)
        return surface

    def _names_pattern(self) -> re.Pattern[str] | None:
        if self._name_re is None and (self._full_names or self._tokens):
            full = [
                r"\s+".join(map(re.escape, _fold(n).split())) for n in self._full_names.values()
            ]
            tokens = [re.escape(_fold(t)) for t in self._tokens]
            alternatives = sorted([*full, *tokens], key=len, reverse=True)
            self._name_re = re.compile(rf"(?<!\w)(?:{'|'.join(alternatives)})(?!\w)", re.IGNORECASE)
        return self._name_re


def _map_strings(value: Any, fn: Callable[[str], str]) -> Any:
    """Apply `fn` to every string reachable through models, lists, tuples and dicts."""
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, BaseModel):
        changes = {}
        for name in type(value).model_fields:
            current = getattr(value, name)
            mapped = _map_strings(current, fn)
            if mapped != current:
                changes[name] = mapped
        return value.model_copy(update=changes) if changes else value
    if isinstance(value, list):
        return [_map_strings(item, fn) for item in value]
    if isinstance(value, tuple):
        return tuple(_map_strings(item, fn) for item in value)
    if isinstance(value, dict):
        return {key: _map_strings(item, fn) for key, item in value.items()}
    return value


def _strings_in(value: Any) -> list[str]:
    found: list[str] = []

    def collect(text: str) -> str:
        found.append(text)
        return text

    _map_strings(value, collect)
    return found


class RedactingGateway:
    """Sends redacted text to `inner` and restores what comes back.

    `redactor` is the configuration template: every call works on `redactor.fresh()`, so
    concurrent calls never share a mapping.
    """

    def __init__(self, inner: LLMGateway, redactor: Redactor | None = None) -> None:
        self._inner = inner
        self._template = redactor or Redactor()

    @property
    def inner(self) -> LLMGateway:
        return self._inner

    @property
    def is_cloud(self) -> bool:
        return self._inner.is_cloud

    @property
    def model(self) -> str:
        return self._inner.model

    def scoped(self, texts: Iterable[str]) -> "RedactingGateway":
        """A gateway whose calls already know the names labelled anywhere in `texts`.

        A packet's documents share people: the name labelled on the claim form is a bare line
        on the licence. Placeholders are still numbered per call.
        """
        template = self._template.fresh()
        template.learn_from(texts)
        return RedactingGateway(self._inner, template)

    @property
    def drift_count(self) -> int:
        """Replay drift of the wrapped gateway, 0 when it is not a replay gateway."""
        return int(getattr(self._inner, "drift_count", 0))

    @property
    def drifted_keys(self) -> list[str]:
        return list(getattr(self._inner, "drifted_keys", []))

    def structured[T: BaseModel](
        self,
        *,
        system: str,
        prompt: str,
        schema: type[T],
        purpose: str,
        cache_key: str | None = None,
    ) -> LLMResult[T]:
        redactor = self._template.fresh()
        redactor.learn(system)
        redactor.learn(prompt)
        safe_system, _ = redactor.redact(system)
        safe_prompt, _ = redactor.redact(prompt)
        result = self._inner.structured(
            system=safe_system,
            prompt=safe_prompt,
            schema=schema,
            purpose=purpose,
            cache_key=cache_key,
        )
        output = redactor.restore(result.output, redactor.mapping)
        return result.model_copy(update={"output": output})

    def chat(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        cache_key: str | None = None,
    ) -> ChatTurn:
        redactor = self._template.fresh()
        # Learn from everything first: a name labelled in a later message must also be
        # removed from the earlier ones.
        for seen in [system, *_strings_in(messages)]:
            redactor.learn(seen)
        safe_system, _ = redactor.redact(system)
        safe_messages = [_redact_message(m, redactor) for m in messages]
        turn = self._inner.chat(
            system=safe_system, messages=safe_messages, tools=tools, cache_key=cache_key
        )
        mapping = redactor.mapping
        text = None if turn.text is None else redactor.restore_text(turn.text, mapping)
        calls = [
            call.model_copy(
                update={
                    "arguments": _map_strings(
                        call.arguments, lambda s: redactor.restore_text(s, mapping)
                    )
                }
            )
            for call in turn.tool_calls
        ]
        return turn.model_copy(update={"text": text, "tool_calls": calls})


def scope_gateway(gateway: LLMGateway, texts: Iterable[str]) -> LLMGateway:
    """`gateway.scoped(texts)` for a gateway that has one, the gateway itself otherwise.

    A redacting gateway scopes its redactor; a wrapper around one (the tracing gateway) scopes
    what it wraps and keeps wrapping. Callers pass the OCR text of every document in a packet
    and never need to know which backend is configured.
    """
    scoped = getattr(gateway, "scoped", None)
    if scoped is None:
        return gateway
    result: LLMGateway = scoped(texts)
    return result


def _redact_message(message: Message, redactor: Redactor) -> Message:
    content, _ = redactor.redact(message.content)
    calls = [
        ToolCall(
            id=call.id,
            name=call.name,
            arguments=_map_strings(call.arguments, lambda s: redactor.redact(s)[0]),
        )
        for call in message.tool_calls
    ]
    return message.model_copy(update={"content": content, "tool_calls": calls})
