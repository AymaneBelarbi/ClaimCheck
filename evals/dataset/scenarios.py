"""Which packets exist, what they contain, and the verdicts they must produce.

Every packet starts from a plan (scenario, claim type, variant). The plan decides the facts that
make the scenario true (an expired licence, an invoice dated before the incident, ...); the seeded
random generator fills in everything else.

Expected rule verdicts are computed twice and must agree. `evaluate_rules` applies the rule
semantics to the gold field values and gold field verdicts, and `intended_rule_verdicts` states
what the scenario is meant to trigger. A disagreement means the generator built a packet that does
not test what it claims to test, and generation stops.
"""

import random
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date, timedelta
from decimal import Decimal
from enum import StrEnum

from faker import Faker
from rapidfuzz import fuzz, utils

from claimcheck.core.types import ClaimType, DocType, Verdict
from evals.dataset.policies import (
    LATEST_INCIDENT,
    POLICIES,
    PolicySpec,
    VersionTerms,
    coverage_limit,
    version_in_force,
)
from evals.dataset.render import (
    DATE_STYLES,
    ClaimFormDoc,
    Document,
    IdCardDoc,
    IncidentReportDoc,
    InvoiceDoc,
)
from evals.dataset.types import (
    FIELD_NAMES,
    NAME_MATCH_THRESHOLD,
    REQUIRED_DOCUMENTS,
    DatasetError,
    GroundTruth,
    GTDocument,
    Scenario,
    rules_for,
)

# Nothing in a packet is dated after this day (the dataset's "today").
LATEST_DOCUMENT = date(2026, 10, 19)

# Margins around the name-match threshold: a "same person" pair must score at least SAME and a
# "different person" pair at most DIFFERENT, under both common preprocessing choices, so the
# expected R1 verdict cannot flip on an implementation detail.
SAME_PERSON_MIN = 95.0
DIFFERENT_PERSON_MAX = 85.0

S = Scenario
C = ClaimType


@dataclass(frozen=True)
class Plan:
    scenario: Scenario
    claim_type: ClaimType
    variant: str | None = None


PLANS: tuple[Plan, ...] = (
    # clean: 16, with passing-side edge cases
    Plan(S.CLEAN, C.COLLISION),
    Plan(S.CLEAN, C.COLLISION),
    Plan(S.CLEAN, C.COLLISION),
    Plan(S.CLEAN, C.COLLISION, "same_day_invoice"),
    Plan(S.CLEAN, C.COLLISION, "incident_on_version_start"),
    Plan(S.CLEAN, C.COLLISION, "total_above_previous_limit"),
    Plan(S.CLEAN, C.GLASS),
    Plan(S.CLEAN, C.GLASS),
    Plan(S.CLEAN, C.GLASS),
    Plan(S.CLEAN, C.GLASS, "total_equals_limit"),
    Plan(S.CLEAN, C.GLASS, "id_expires_on_incident"),
    Plan(S.CLEAN, C.THEFT),
    Plan(S.CLEAN, C.THEFT),
    Plan(S.CLEAN, C.THEFT),
    Plan(S.CLEAN, C.THEFT),
    Plan(S.CLEAN, C.THEFT, "id_expires_on_incident"),
    # name_mismatch: 4
    Plan(S.NAME_MISMATCH, C.COLLISION),
    Plan(S.NAME_MISMATCH, C.GLASS),
    Plan(S.NAME_MISMATCH, C.THEFT),
    Plan(S.NAME_MISMATCH, C.COLLISION),
    # incident_outside_coverage: 4
    Plan(S.INCIDENT_OUTSIDE_COVERAGE, C.COLLISION, "on_lapse_date"),
    Plan(S.INCIDENT_OUTSIDE_COVERAGE, C.THEFT, "before_first_version"),
    Plan(S.INCIDENT_OUTSIDE_COVERAGE, C.THEFT, "after_lapse"),
    Plan(S.INCIDENT_OUTSIDE_COVERAGE, C.COLLISION, "day_before_first_version"),
    # id_expired: 4
    Plan(S.ID_EXPIRED, C.COLLISION, "expired_day_before"),
    Plan(S.ID_EXPIRED, C.THEFT),
    Plan(S.ID_EXPIRED, C.GLASS),
    Plan(S.ID_EXPIRED, C.COLLISION),
    # missing_document: 6, two per missing type
    Plan(S.MISSING_DOCUMENT, C.COLLISION, "missing:id_card"),
    Plan(S.MISSING_DOCUMENT, C.THEFT, "missing:id_card"),
    Plan(S.MISSING_DOCUMENT, C.COLLISION, "missing:invoice"),
    Plan(S.MISSING_DOCUMENT, C.GLASS, "missing:invoice"),
    Plan(S.MISSING_DOCUMENT, C.COLLISION, "missing:incident_report"),
    Plan(S.MISSING_DOCUMENT, C.THEFT, "missing:incident_report"),
    # unreadable_scan: 5; only documents that feed a field-based rule are degraded
    Plan(S.UNREADABLE_SCAN, C.COLLISION, "unreadable:claim_form"),
    Plan(S.UNREADABLE_SCAN, C.THEFT, "unreadable:id_card"),
    Plan(S.UNREADABLE_SCAN, C.GLASS, "unreadable:invoice"),
    Plan(S.UNREADABLE_SCAN, C.COLLISION, "unreadable:invoice"),
    Plan(S.UNREADABLE_SCAN, C.GLASS, "unreadable:id_card"),
    # total_over_limit: 4
    Plan(S.TOTAL_OVER_LIMIT, C.COLLISION),
    Plan(S.TOTAL_OVER_LIMIT, C.COLLISION),
    Plan(S.TOTAL_OVER_LIMIT, C.GLASS),
    Plan(S.TOTAL_OVER_LIMIT, C.GLASS, "one_cent_over"),
    # invoice_before_incident: 4
    Plan(S.INVOICE_BEFORE_INCIDENT, C.COLLISION, "day_before"),
    Plan(S.INVOICE_BEFORE_INCIDENT, C.GLASS),
    Plan(S.INVOICE_BEFORE_INCIDENT, C.COLLISION),
    Plan(S.INVOICE_BEFORE_INCIDENT, C.GLASS),
    # duplicate_conflicting_invoice: 3
    Plan(S.DUPLICATE_CONFLICTING_INVOICE, C.COLLISION),
    Plan(S.DUPLICATE_CONFLICTING_INVOICE, C.GLASS),
    Plan(S.DUPLICATE_CONFLICTING_INVOICE, C.COLLISION),
)


@dataclass(frozen=True)
class PacketFile:
    filename: str
    doc: Document
    unreadable: bool


@dataclass(frozen=True)
class Packet:
    packet_id: str
    plan: Plan
    policy: PolicySpec
    incident_date: date
    files: tuple[PacketFile, ...]
    truth: GroundTruth


# --- names -------------------------------------------------------------------------------


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def name_scores(a: str, b: str) -> tuple[float, float]:
    """R1 similarity with and without punctuation/whitespace preprocessing."""
    left, right = strip_accents(a).casefold(), strip_accents(b).casefold()
    return (
        fuzz.token_sort_ratio(left, right),
        fuzz.token_sort_ratio(left, right, processor=utils.default_process),
    )


def names_match(a: str, b: str) -> bool:
    return name_scores(a, b)[0] >= NAME_MATCH_THRESHOLD


_LOCALES = ("en_US", "en_GB", "es_ES", "fr_FR", "de_DE", "pt_PT", "it_IT")

# How the licence prints the holder's name. Every variant is the same person for R1.
_ID_NAME_STYLES = ("as_written", "surname_first_upper", "upper", "no_accents")


def _latin1(text: str) -> bool:
    try:
        text.encode("latin-1")
    except UnicodeEncodeError:
        return False
    return True


class _People:
    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self._fakers: dict[str, Faker] = {}
        for locale in _LOCALES:
            faker = Faker(locale)
            faker.seed_instance(rng.getrandbits(32))
            self._fakers[locale] = faker
        self.us = self._fakers["en_US"]

    def person(self) -> tuple[str, str, str]:
        while True:
            locale = self._rng.choice(_LOCALES)
            faker = self._fakers[locale]
            first, last = faker.first_name(), faker.last_name()
            if _latin1(first + last) and " " not in first:
                return locale, first, last

    def other_first_name(self, locale: str, first: str, last: str) -> str:
        """A different first name that clearly fails the name match against `first last`."""
        faker = self._fakers[locale]
        for _ in range(200):
            candidate = faker.first_name()
            if " " in candidate or not _latin1(candidate):
                continue
            scores = name_scores(f"{first} {last}", f"{candidate} {last}")
            if max(scores) <= DIFFERENT_PERSON_MAX:
                return candidate
        raise DatasetError(f"no clearly different first name found for {first} {last}")


def _id_name(first: str, last: str, style: str) -> str:
    if style == "surname_first_upper":
        return f"{last} {first}".upper()
    if style == "upper":
        return f"{first} {last}".upper()
    if style == "no_accents":
        return strip_accents(f"{first} {last}")
    return f"{first} {last}"


# --- dates and money ---------------------------------------------------------------------


def _days(n: int) -> timedelta:
    return timedelta(days=n)


def _day_in(rng: random.Random, v: VersionTerms) -> date:
    end = min(v.effective_to or date.max, LATEST_INCIDENT + _days(1))
    return v.effective_from + _days(rng.randrange((end - v.effective_from).days))


def _shift_years(day: date, years: int) -> date:
    try:
        return day.replace(year=day.year + years)
    except ValueError:  # 29 February
        return day.replace(year=day.year + years, day=28)


def _cents(rng: random.Random, low: Decimal, high: Decimal) -> Decimal:
    return Decimal(rng.randrange(int(low * 100), int(high * 100) + 1)) / 100


_COLLISION_ITEMS = (
    "Front bumper cover, replace",
    "Rear bumper cover, replace",
    "Headlamp assembly",
    "Bonnet, repair and refinish",
    "Wing panel, replace",
    "Paint and refinish materials",
    "Radiator support",
    "Parking sensor, replace",
    "Wheel alignment",
    "Tail lamp assembly",
)
_GLASS_ITEMS = (
    "Windscreen, OEM equivalent",
    "Windscreen moulding kit",
    "Urethane adhesive and primer",
    "ADAS camera recalibration",
    "Rain sensor gel pad",
    "Side window glass",
)


def _items(
    rng: random.Random, claim_type: ClaimType, total: Decimal
) -> tuple[tuple[str, Decimal], ...]:
    """Line items that add up exactly to `total`, labour last."""
    if claim_type == ClaimType.GLASS:
        names = rng.sample(_GLASS_ITEMS, rng.randint(1, 3))
        hours = rng.choice(("1.0", "1.5", "2.0"))
    else:
        names = rng.sample(_COLLISION_ITEMS, rng.randint(2, 4))
        hours = rng.choice(("4.5", "6.0", "7.5", "9.0", "12.0"))
    names.append(f"Labour ({hours} hours)")
    weights = [rng.uniform(1.0, 4.0) for _ in names]
    # Every line gets at least $1.00; the rest is shared by weight, labour takes the rounding.
    spare = int(total * 100) - 100 * len(names)
    if spare < 0:
        raise DatasetError(f"total {total} too small for {len(names)} line items")
    shares = [100 + int(spare * w / sum(weights)) for w in weights]
    shares[-1] = int(total * 100) - sum(shares[:-1])
    return tuple((n, Decimal(s) / 100) for n, s in zip(names, shares, strict=True))


# --- packet facts ------------------------------------------------------------------------


@dataclass(frozen=True)
class _Facts:
    policy: PolicySpec
    version: VersionTerms | None
    incident_date: date


def _pick_version(
    rng: random.Random, claim_type: ClaimType, *, min_version: int = 1
) -> tuple[PolicySpec, VersionTerms]:
    eligible = [
        (p, v)
        for p in POLICIES
        for v in p.versions
        if v.version_no >= min_version
        and v.effective_from <= LATEST_INCIDENT
        and (claim_type != ClaimType.GLASS or v.glass_limit is not None)
    ]
    return rng.choice(eligible)


def _lapsed_policy() -> PolicySpec:
    for p in POLICIES:
        if p.versions[-1].effective_to is not None:
            return p
    raise DatasetError("no lapsed policy defined")


def _facts(rng: random.Random, plan: Plan) -> _Facts:
    variant = plan.variant
    if plan.scenario == S.INCIDENT_OUTSIDE_COVERAGE:
        if variant in ("on_lapse_date", "after_lapse"):
            policy = _lapsed_policy()
            end = policy.versions[-1].effective_to or date.max
            gap = 0 if variant == "on_lapse_date" else rng.randint(10, 150)
            return _Facts(policy, None, end + _days(gap))
        policy = rng.choice(POLICIES)
        gap = 1 if variant == "day_before_first_version" else rng.randint(10, 200)
        return _Facts(policy, None, policy.versions[0].effective_from - _days(gap))
    if variant == "incident_on_version_start":
        policy, version = _pick_version(rng, plan.claim_type, min_version=2)
        return _Facts(policy, version, version.effective_from)
    if variant == "total_above_previous_limit":
        # A collision total that only the raised limit of AUT-104233 v3 covers.
        policy = POLICIES[0]
        version = policy.versions[2]
        return _Facts(policy, version, _day_in(rng, version))
    policy, version = _pick_version(rng, plan.claim_type)
    return _Facts(policy, version, _day_in(rng, version))


def _total(rng: random.Random, plan: Plan, facts: _Facts) -> Decimal:
    claim_type = plan.claim_type
    limit = coverage_limit(facts.version, claim_type) if facts.version else None
    if plan.variant == "total_above_previous_limit":
        return _cents(rng, Decimal("15500"), Decimal("22000"))
    if plan.scenario == S.TOTAL_OVER_LIMIT:
        if limit is None:
            raise DatasetError(f"{plan}: no limit to exceed")
        if plan.variant == "one_cent_over":
            return limit + Decimal("0.01")
        over = (40, 600) if claim_type == ClaimType.GLASS else (250, 4000)
        return limit + _cents(rng, Decimal(over[0]), Decimal(over[1]))
    if plan.variant == "total_equals_limit":
        if limit is None:
            raise DatasetError(f"{plan}: no limit to match")
        return limit
    if claim_type == ClaimType.GLASS:
        ceiling = (limit or Decimal("1000")) * Decimal("0.6")
        return _cents(rng, Decimal("160"), ceiling)
    ceiling = min((limit or Decimal("10000")) * Decimal("0.6"), Decimal("9000"))
    return _cents(rng, Decimal("850"), ceiling)


def _invoice_date(rng: random.Random, plan: Plan, incident: date) -> date:
    if plan.variant == "same_day_invoice":
        return incident
    if plan.scenario == S.INVOICE_BEFORE_INCIDENT:
        return incident - _days(1 if plan.variant == "day_before" else rng.randint(2, 30))
    return min(incident + _days(rng.randint(1, 25)), LATEST_DOCUMENT)


def _expiry(rng: random.Random, plan: Plan, incident: date) -> date:
    if plan.variant == "id_expires_on_incident":
        return incident
    if plan.scenario == S.ID_EXPIRED:
        return incident - _days(1 if plan.variant == "expired_day_before" else rng.randint(5, 400))
    return incident + _days(rng.randint(120, 2500))


def _plate(rng: random.Random) -> str:
    letters = "ABCEFGHJKLMNPRSTUVWXYZ"
    return "".join(rng.choice(letters) for _ in range(3)) + f"-{rng.randrange(10000):04d}"


_VEHICLES = (
    "2019 Toyota Corolla",
    "2021 Honda Civic",
    "2018 Ford Focus",
    "2020 Volkswagen Golf",
    "2022 Hyundai Tucson",
    "2017 Mazda 3",
    "2023 Kia Sportage",
    "2016 Subaru Outback",
)


# What the claimant wrote and what the police recorded, as matching pairs so the two documents in
# a packet describe the same event. Glass claims have no police report.
_STORIES: dict[ClaimType, tuple[tuple[str, str], ...]] = {
    ClaimType.COLLISION: (
        (
            "I was stopped at the lights on {street} when the car behind did not brake in time "
            "and hit my rear bumper.",
            "Two-vehicle collision on {street}. Vehicle {plate} was struck from behind while "
            "stationary at a red signal. No injuries reported. Both drivers exchanged details.",
        ),
        (
            "Another driver pulled out of a side road onto {street} and hit the front left of my "
            "car. Nobody was hurt.",
            "Vehicle {plate} was struck on the front nearside by a second vehicle emerging from "
            "a side road onto {street}. Minor damage to both vehicles, no injuries. Road was dry.",
        ),
        (
            "While reversing out of a parking space on {street} I hit a concrete bollard that I "
            "had not seen.",
            "Single-vehicle incident in a car park on {street}. Driver of {plate} reports "
            "striking a bollard at low speed while reversing. No injuries.",
        ),
        (
            "A deer ran across {street} after dark and I could not avoid it. The front of the "
            "car is damaged.",
            "Vehicle {plate} struck a deer on {street} at approximately {time}. Animal did not "
            "survive. Driver uninjured. Front-end damage, vehicle driven from scene.",
        ),
    ),
    ClaimType.GLASS: (
        (
            "A stone thrown up by a truck on {street} hit the windscreen and the crack has "
            "spread across the driver's side.",
            "",
        ),
        (
            "Driving on {street} a piece of gravel chipped the windscreen. The chip has now "
            "cracked and needs a new screen.",
            "",
        ),
        (
            "The windscreen cracked overnight while the car was parked on {street} during the "
            "cold snap.",
            "",
        ),
    ),
    ClaimType.THEFT: (
        (
            "I parked the car on {street} in the evening and it was gone the next morning. I "
            "still have both keys.",
            "Owner states vehicle {plate} was left locked on {street} at {time} and found "
            "missing the following morning. No witnesses. Owner holds both keys. Vehicle "
            "entered on the stolen vehicle register.",
        ),
        (
            "The car was taken from outside my home on {street} overnight. I reported it to the "
            "police as soon as I noticed.",
            "Vehicle {plate} reported stolen from outside the owner's residence on {street} "
            "between {time} and 07:00. No signs of broken glass at the scene. Vehicle entered "
            "on the stolen vehicle register.",
        ),
        (
            "My car was stolen from a car park off {street} while I was at work. Both keys are "
            "with me.",
            "Vehicle {plate} taken from a pay-and-display car park off {street} during the "
            "working day. Owner holds both keys. CCTV request made to the car park operator.",
        ),
    ),
}


def _story(rng: random.Random, claim_type: ClaimType, street: str, plate: str) -> tuple[str, str]:
    claim, report = rng.choice(_STORIES[claim_type])
    time = f"{rng.randint(19, 23)}:{rng.choice(('00', '15', '30', '45'))}"
    return (
        claim.format(street=street, plate=plate, time=time),
        report.format(street=street, plate=plate, time=time),
    )


def _vendor(people: _People, rng: random.Random, claim_type: ClaimType, city: str) -> str:
    last = people.us.last_name()
    if claim_type == ClaimType.GLASS:
        return rng.choice((f"{city} Auto Glass", f"{last} Windscreen Services", "ClearView Glass"))
    return rng.choice(
        (f"{last} Auto Body", f"{city} Collision Centre", f"{last} & Sons Panelworks")
    )


def _style(rng: random.Random) -> str:
    return rng.choice(DATE_STYLES)


# --- ground truth ------------------------------------------------------------------------


def _gt_documents(files: tuple[PacketFile, ...]) -> list[GTDocument]:
    by_type: dict[DocType, list[PacketFile]] = defaultdict(list)
    for f in files:
        by_type[f.doc.doc_type].append(f)
    out = []
    for f in files:
        gold: dict[str, str | None] = dict(f.doc.gold())
        if f.unreadable:
            verdicts = dict.fromkeys(gold, Verdict.LOW_CONFIDENCE)
        else:
            verdicts = dict.fromkeys(gold, Verdict.FOUND)
            # Two documents of the same type that disagree on a field: neither value can be
            # trusted, so the field is conflicting on both.
            for name in gold:
                if len({g.doc.gold()[name] for g in by_type[f.doc.doc_type]}) > 1:
                    verdicts[name] = Verdict.CONFLICTING
        if list(gold) != list(FIELD_NAMES[f.doc.doc_type]):
            raise DatasetError(f"{f.filename}: fields {list(gold)} do not match the schema")
        out.append(
            GTDocument(
                filename=f.filename, doc_type=f.doc.doc_type, fields=gold, field_verdicts=verdicts
            )
        )
    return out


class _Status(StrEnum):
    OK = "ok"
    MISSING = "missing"
    LOW = "low"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class _Input:
    status: _Status
    value: str | None = None


def _read(docs: list[GTDocument], doc_type: DocType, field: str) -> _Input:
    present = [d for d in docs if d.doc_type == doc_type]
    if not present:
        return _Input(_Status.MISSING)
    verdicts = {d.field_verdicts[field] for d in present}
    values = {d.fields[field] for d in present}
    if Verdict.LOW_CONFIDENCE in verdicts:
        return _Input(_Status.LOW)
    if Verdict.MISSING in verdicts or None in values:
        return _Input(_Status.MISSING)
    if Verdict.CONFLICTING in verdicts or len(values) > 1:
        return _Input(_Status.CONFLICT)
    return _Input(_Status.OK, values.pop())


_BLOCKED = {
    _Status.MISSING: Verdict.MISSING,
    _Status.LOW: Verdict.LOW_CONFIDENCE,
    _Status.CONFLICT: Verdict.CONFLICTING,
}


def _blocked(rule_id: str, *inputs: _Input) -> Verdict | None:
    problems = {i.status for i in inputs} - {_Status.OK}
    if len(problems) > 1:
        # The verdict would depend on which check a rule runs first. The dataset avoids that.
        raise DatasetError(f"{rule_id}: inputs are {sorted(problems)}; expected verdict ambiguous")
    return _BLOCKED[problems.pop()] if problems else None


def _passes(ok: bool) -> Verdict:
    return Verdict.FOUND if ok else Verdict.CONFLICTING


def evaluate_rules(
    claim_type: ClaimType,
    incident_date: date,
    version: VersionTerms | None,
    docs: list[GTDocument],
) -> dict[str, Verdict]:
    """Reference semantics of R1-R7 applied to gold fields and gold field verdicts."""
    out: dict[str, Verdict] = {}
    for rule_id in rules_for(claim_type):
        if rule_id == "R1_NAME_MATCH":
            claimant = _read(docs, DocType.CLAIM_FORM, "claimant_name")
            holder = _read(docs, DocType.ID_CARD, "full_name")
            out[rule_id] = _blocked(rule_id, claimant, holder) or _passes(
                names_match(claimant.value or "", holder.value or "")
            )
        elif rule_id == "R2_INCIDENT_IN_COVERAGE":
            out[rule_id] = _passes(version is not None)
        elif rule_id == "R3_INVOICE_WITHIN_LIMIT":
            total = _read(docs, DocType.INVOICE, "total")
            limit = coverage_limit(version, claim_type) if version else None
            limit_in = (
                _Input(_Status.OK, str(limit)) if limit is not None else _Input(_Status.MISSING)
            )
            out[rule_id] = _blocked(rule_id, total, limit_in) or _passes(
                Decimal(total.value or "0") <= Decimal(limit_in.value or "0")
            )
        elif rule_id == "R4_INVOICE_AFTER_INCIDENT":
            issued = _read(docs, DocType.INVOICE, "invoice_date")
            out[rule_id] = _blocked(rule_id, issued) or _passes(
                date.fromisoformat(issued.value or "") >= incident_date
            )
        elif rule_id == "R5_ID_VALID_ON_INCIDENT":
            expiry = _read(docs, DocType.ID_CARD, "expiry_date")
            out[rule_id] = _blocked(rule_id, expiry) or _passes(
                date.fromisoformat(expiry.value or "") >= incident_date
            )
        elif rule_id == "R6_REQUIRED_DOCUMENTS":
            present = {d.doc_type for d in docs}
            ok = REQUIRED_DOCUMENTS[claim_type] <= present
            out[rule_id] = Verdict.FOUND if ok else Verdict.MISSING
        elif rule_id == "R7_INCIDENT_DATE_CONSISTENT":
            stated = [_read(docs, DocType.CLAIM_FORM, "incident_date")]
            # The incident report is optional for this rule: compared only when it exists.
            if any(d.doc_type == DocType.INCIDENT_REPORT for d in docs):
                stated.append(_read(docs, DocType.INCIDENT_REPORT, "incident_date"))
            out[rule_id] = _blocked(rule_id, *stated) or _passes(
                all(date.fromisoformat(s.value or "") == incident_date for s in stated)
            )
        else:
            raise DatasetError(f"no reference semantics for {rule_id}")
    return out


def intended_rule_verdicts(plan: Plan) -> dict[str, Verdict]:
    """What each scenario is designed to trigger; every other applicable rule passes."""
    out = dict.fromkeys(rules_for(plan.claim_type), Verdict.FOUND)
    has_invoice_rules = plan.claim_type != ClaimType.THEFT
    s, v = plan.scenario, plan.variant
    if s == S.NAME_MISMATCH:
        out["R1_NAME_MATCH"] = Verdict.CONFLICTING
    elif s == S.INCIDENT_OUTSIDE_COVERAGE:
        out["R2_INCIDENT_IN_COVERAGE"] = Verdict.CONFLICTING
        if has_invoice_rules:
            out["R3_INVOICE_WITHIN_LIMIT"] = Verdict.MISSING
    elif s == S.ID_EXPIRED:
        out["R5_ID_VALID_ON_INCIDENT"] = Verdict.CONFLICTING
    elif s == S.MISSING_DOCUMENT:
        out["R6_REQUIRED_DOCUMENTS"] = Verdict.MISSING
        if v == "missing:id_card":
            out["R1_NAME_MATCH"] = out["R5_ID_VALID_ON_INCIDENT"] = Verdict.MISSING
        elif v == "missing:invoice":
            out["R3_INVOICE_WITHIN_LIMIT"] = out["R4_INVOICE_AFTER_INCIDENT"] = Verdict.MISSING
    elif s == S.UNREADABLE_SCAN:
        low = Verdict.LOW_CONFIDENCE
        if v == "unreadable:claim_form":
            out["R1_NAME_MATCH"] = out["R7_INCIDENT_DATE_CONSISTENT"] = low
        elif v == "unreadable:id_card":
            out["R1_NAME_MATCH"] = out["R5_ID_VALID_ON_INCIDENT"] = low
        elif v == "unreadable:invoice":
            out["R3_INVOICE_WITHIN_LIMIT"] = out["R4_INVOICE_AFTER_INCIDENT"] = low
    elif s in (S.TOTAL_OVER_LIMIT, S.DUPLICATE_CONFLICTING_INVOICE):
        out["R3_INVOICE_WITHIN_LIMIT"] = Verdict.CONFLICTING
    elif s == S.INVOICE_BEFORE_INCIDENT:
        out["R4_INVOICE_AFTER_INCIDENT"] = Verdict.CONFLICTING
    return out


# --- packet assembly ---------------------------------------------------------------------


def _check_plan(plan: Plan) -> None:
    """Reject plans whose claim type cannot carry the scenario."""
    needs_invoice = plan.scenario in (
        S.TOTAL_OVER_LIMIT,
        S.INVOICE_BEFORE_INCIDENT,
        S.DUPLICATE_CONFLICTING_INVOICE,
    ) or plan.variant in (
        "missing:invoice",
        "unreadable:invoice",
        "total_equals_limit",
        "same_day_invoice",
    )
    needs_report = plan.variant == "missing:incident_report"
    # Glass claims are only filed against versions with glass cover, so never outside coverage.
    no_glass = plan.scenario == S.INCIDENT_OUTSIDE_COVERAGE
    ok = not (
        (needs_invoice and plan.claim_type == C.THEFT)
        or ((needs_report or no_glass) and plan.claim_type == C.GLASS)
        or (plan.variant == "total_above_previous_limit" and plan.claim_type != C.COLLISION)
    )
    if not ok:
        raise DatasetError(f"{plan} cannot be built for a {plan.claim_type} claim")


def build_packet(
    rng: random.Random, people: _People, packet_id: str, plan: Plan, name_style: str
) -> Packet:
    _check_plan(plan)
    facts = _facts(rng, plan)
    incident = facts.incident_date
    if plan.scenario != S.INCIDENT_OUTSIDE_COVERAGE and facts.version is None:
        raise DatasetError(f"{packet_id}: no version in force on {incident}")
    if version_in_force(facts.policy, incident) is not facts.version:
        raise DatasetError(f"{packet_id}: version bookkeeping is wrong for {incident}")

    locale, first, last = people.person()
    claimant = f"{first} {last}"
    if plan.scenario == S.NAME_MISMATCH:
        other = people.other_first_name(locale, first, last)
        printed_id_name = _id_name(other, last, name_style)
    else:
        printed_id_name = _id_name(first, last, name_style)
    lowest, highest = sorted(name_scores(claimant, printed_id_name))
    if plan.scenario == S.NAME_MISMATCH and highest > DIFFERENT_PERSON_MAX:
        raise DatasetError(f"{packet_id}: mismatch names too close ({highest})")
    if plan.scenario != S.NAME_MISMATCH and lowest < SAME_PERSON_MIN:
        raise DatasetError(f"{packet_id}: same-person names too far apart ({lowest})")

    us = people.us
    city = us.city()
    street = us.street_name()
    plate = _plate(rng)
    claim_type = plan.claim_type
    claim_story, report_story = _story(rng, claim_type, street, plate)

    docs: list[Document] = [
        ClaimFormDoc(
            claimant_name=claimant,
            policy_number=facts.policy.policy_number,
            incident_date=incident,
            incident_type=claim_type,
            vehicle_plate=plate,
            vehicle=rng.choice(_VEHICLES),
            phone=f"(555) {rng.randrange(200, 1000)}-{rng.randrange(10000):04d}",
            description=claim_story,
            signed_on=min(incident + _days(rng.randint(1, 20)), LATEST_DOCUMENT),
            layout=rng.randrange(2),
            date_style=_style(rng),
        )
    ]

    expiry = _expiry(rng, plan, incident)
    birth = date(rng.randint(1952, 2002), rng.randint(1, 12), rng.randint(1, 28))
    docs.append(
        IdCardDoc(
            full_name=printed_id_name,
            id_number=f"D{rng.randrange(10**8):08d}",
            date_of_birth=birth,
            expiry_date=expiry,
            issue_date=_shift_years(expiry, -8),
            address=f"{us.building_number()} {us.street_name()}, {us.city()}",
            licence_class=rng.choice(("B", "C", "D")),
            layout=rng.randrange(2),
            date_style=_style(rng),
        )
    )

    if claim_type != ClaimType.THEFT:
        total = _total(rng, plan, facts)
        if plan.scenario == S.DUPLICATE_CONFLICTING_INVOICE:
            total = _cents(rng, Decimal("160"), total)
        invoice = InvoiceDoc(
            vendor=_vendor(people, rng, claim_type, city),
            vendor_address=f"{us.building_number()} {us.street_name()}, {city}",
            vendor_phone=f"(555) {rng.randrange(200, 1000)}-{rng.randrange(10000):04d}",
            invoice_number=f"INV-{rng.randrange(10000, 100000)}",
            invoice_date=_invoice_date(rng, plan, incident),
            bill_to=claimant,
            vehicle_plate=plate,
            items=_items(rng, claim_type, total),
            layout=rng.randrange(2),
            date_style=_style(rng),
        )
        docs.append(invoice)
        if plan.scenario == S.DUPLICATE_CONFLICTING_INVOICE:
            # The same invoice sent again with one line raised: same number, vendor and date.
            bump = _cents(rng, Decimal("120"), Decimal("900"))
            items = list(invoice.items)
            k = rng.randrange(len(items))
            items[k] = (items[k][0], items[k][1] + bump)
            docs.append(
                replace(
                    invoice,
                    items=tuple(items),
                    layout=1 - invoice.layout,
                    date_style=_style(rng),
                )
            )

    if claim_type != ClaimType.GLASS:
        docs.append(
            IncidentReportDoc(
                agency=f"{city} Police Department",
                report_number=f"IR-{incident.year}-{rng.randrange(10**6):06d}",
                incident_date=incident,
                report_date=min(incident + _days(rng.randint(0, 3)), LATEST_DOCUMENT),
                location=f"{us.building_number()} {street}, {city}",
                officer=f"Officer {us.last_name()} (badge {rng.randrange(1000, 10000)})",
                vehicle_plate=plate,
                description=report_story,
                layout=rng.randrange(2),
                date_style=_style(rng),
            )
        )

    kind, _, target = (plan.variant or "").partition(":")
    missing = target if kind == "missing" else None
    unreadable = target if kind == "unreadable" else None
    docs = [d for d in docs if d.doc_type.value != missing]
    if missing and len(docs) == len(REQUIRED_DOCUMENTS[claim_type]):
        raise DatasetError(f"{packet_id}: {missing} was not part of a {claim_type} packet")
    rng.shuffle(docs)

    files = []
    for i, doc in enumerate(docs, start=1):
        is_unreadable = doc.doc_type.value == unreadable
        ext = "png" if is_unreadable or isinstance(doc, IdCardDoc) else "pdf"
        files.append(PacketFile(f"doc_{i}.{ext}", doc, is_unreadable))
    if unreadable and sum(f.unreadable for f in files) != 1:
        raise DatasetError(f"{packet_id}: expected exactly one unreadable {unreadable}")

    gt_docs = _gt_documents(tuple(files))
    verdicts = evaluate_rules(claim_type, incident, facts.version, gt_docs)
    intended = intended_rule_verdicts(plan)
    if verdicts != intended:
        raise DatasetError(
            f"{packet_id} ({plan}): rules give {verdicts}, scenario wants {intended}"
        )

    truth = GroundTruth(
        packet_id=packet_id,
        policy_number=facts.policy.policy_number,
        holder_id=facts.policy.holder_id,
        claim_type=claim_type,
        incident_date=incident,
        scenario=plan.scenario,
        variant=plan.variant,
        documents=gt_docs,
        rule_verdicts=verdicts,
    )
    return Packet(packet_id, plan, facts.policy, incident, tuple(files), truth)


def build_packets(rng: random.Random) -> list[Packet]:
    people = _People(rng)
    plans = list(PLANS)
    # Packet ids must not reveal the scenario.
    rng.shuffle(plans)
    packets = []
    for i, plan in enumerate(plans, start=1):
        # Rotate the printed-name style so clean packets cover every variant.
        style = _ID_NAME_STYLES[i % len(_ID_NAME_STYLES)]
        packets.append(build_packet(rng, people, f"PKT-{i:03d}", plan, style))
    return packets


def split(packets: list[Packet], rng: random.Random, holdout: int) -> dict[str, list[str]]:
    """Stratified by scenario, so every failure mode appears in both splits."""
    groups: dict[Scenario, list[str]] = defaultdict(list)
    for p in packets:
        groups[p.plan.scenario].append(p.packet_id)
    share = holdout / len(packets)
    exact = {s: len(ids) * share for s, ids in groups.items()}
    quota = {s: max(1, int(x)) for s, x in exact.items()}
    # Hand out the remaining places to the scenarios furthest below their proportional share.
    by_deficit = sorted(groups, key=lambda s: (quota[s] - exact[s], s.value))
    for s in by_deficit:
        if sum(quota.values()) >= holdout:
            break
        quota[s] += 1
    if sum(quota.values()) != holdout or any(quota[s] >= len(groups[s]) for s in groups):
        raise DatasetError(f"cannot stratify {holdout} holdout packets: {quota}")
    held: list[str] = []
    for s in sorted(groups, key=lambda s: s.value):
        ids = sorted(groups[s])
        rng.shuffle(ids)
        held += ids[: quota[s]]
    dev = sorted(p.packet_id for p in packets if p.packet_id not in held)
    return {"dev": dev, "holdout": sorted(held)}
