"""The three synthetic motor policies and their versions, written as code.

Each version is rendered to markdown with one `## <ref> <title>` section per clause. Only the
cover clauses carry "collision", "glass" or "theft" in their title and only they state a
`limit of $<amount> per claim`, so the title keyword and the limit pattern pick out exactly one
clause per coverage type.

What changes between versions is deliberate: glass cover starts in v2, the collision limit rises
in v3 of AUT-104233, theft excess changes, and a hire-car benefit replaces a hire-car exclusion.
AUT-311590 lapses (its last version has an end date and no successor).
"""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from claimcheck.core.types import ClaimType

INSURER = "Northgate Mutual"

# The latest incident date used anywhere in the dataset; open-ended versions are sampled up to it.
LATEST_INCIDENT = date(2026, 9, 20)


@dataclass(frozen=True)
class VersionTerms:
    version_no: int
    effective_from: date
    effective_to: date | None
    collision_limit: Decimal
    theft_limit: Decimal
    glass_limit: Decimal | None  # None: glass is excluded in this version
    collision_excess: Decimal
    theft_excess: Decimal
    glass_excess: Decimal | None
    notice_days: int
    hire_car_daily: Decimal | None  # None: replacement vehicles are excluded


@dataclass(frozen=True)
class PolicySpec:
    policy_number: str
    holder_id: str
    versions: tuple[VersionTerms, ...]


@dataclass(frozen=True)
class ClauseDef:
    ref: str
    title: str
    text: str


def _d(value: str) -> Decimal:
    return Decimal(value)


POLICIES: tuple[PolicySpec, ...] = (
    PolicySpec(
        policy_number="AUT-104233",
        holder_id="cust-1",
        versions=(
            VersionTerms(
                version_no=1,
                effective_from=date(2024, 1, 1),
                effective_to=date(2025, 1, 1),
                collision_limit=_d("15000"),
                theft_limit=_d("20000"),
                glass_limit=None,
                collision_excess=_d("500"),
                theft_excess=_d("500"),
                glass_excess=None,
                notice_days=30,
                hire_car_daily=None,
            ),
            VersionTerms(
                version_no=2,
                effective_from=date(2025, 1, 1),
                effective_to=date(2026, 1, 1),
                collision_limit=_d("15000"),
                theft_limit=_d("20000"),
                glass_limit=_d("1500"),
                collision_excess=_d("500"),
                theft_excess=_d("500"),
                glass_excess=_d("100"),
                notice_days=30,
                hire_car_daily=None,
            ),
            VersionTerms(
                version_no=3,
                effective_from=date(2026, 1, 1),
                effective_to=None,
                collision_limit=_d("25000"),
                theft_limit=_d("20000"),
                glass_limit=_d("1500"),
                collision_excess=_d("500"),
                theft_excess=_d("750"),
                glass_excess=_d("100"),
                notice_days=45,
                hire_car_daily=_d("40"),
            ),
        ),
    ),
    PolicySpec(
        policy_number="AUT-208817",
        holder_id="cust-2",
        versions=(
            VersionTerms(
                version_no=1,
                effective_from=date(2024, 3, 15),
                effective_to=date(2025, 3, 15),
                collision_limit=_d("12000"),
                theft_limit=_d("18000"),
                glass_limit=None,
                collision_excess=_d("400"),
                theft_excess=_d("400"),
                glass_excess=None,
                notice_days=30,
                hire_car_daily=None,
            ),
            VersionTerms(
                version_no=2,
                effective_from=date(2025, 3, 15),
                effective_to=None,
                collision_limit=_d("12000"),
                theft_limit=_d("18000"),
                glass_limit=_d("1200"),
                collision_excess=_d("400"),
                theft_excess=_d("600"),
                glass_excess=_d("75"),
                notice_days=30,
                hire_car_daily=_d("35"),
            ),
        ),
    ),
    PolicySpec(
        policy_number="AUT-311590",
        holder_id="cust-3",
        versions=(
            VersionTerms(
                version_no=1,
                effective_from=date(2024, 6, 1),
                effective_to=date(2025, 6, 1),
                collision_limit=_d("10000"),
                theft_limit=_d("15000"),
                glass_limit=None,
                collision_excess=_d("300"),
                theft_excess=_d("300"),
                glass_excess=None,
                notice_days=28,
                hire_car_daily=None,
            ),
            VersionTerms(
                version_no=2,
                effective_from=date(2025, 6, 1),
                # Lapsed for non-payment; no later version exists.
                effective_to=date(2026, 3, 1),
                collision_limit=_d("14000"),
                theft_limit=_d("15000"),
                glass_limit=_d("1000"),
                collision_excess=_d("300"),
                theft_excess=_d("450"),
                glass_excess=_d("50"),
                notice_days=28,
                hire_car_daily=None,
            ),
        ),
    ),
)


def get_policy(policy_number: str) -> PolicySpec:
    for policy in POLICIES:
        if policy.policy_number == policy_number:
            return policy
    raise KeyError(policy_number)


def version_in_force(policy: PolicySpec, day: date) -> VersionTerms | None:
    """Half-open ranges, as in the database: effective_from <= day < effective_to."""
    for v in policy.versions:
        if v.effective_from <= day and (v.effective_to is None or day < v.effective_to):
            return v
    return None


def coverage_limit(version: VersionTerms, claim_type: ClaimType) -> Decimal | None:
    return {
        ClaimType.COLLISION: version.collision_limit,
        ClaimType.GLASS: version.glass_limit,
        ClaimType.THEFT: version.theft_limit,
    }[claim_type]


_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def _long_date(day: date) -> str:
    return f"{day.day} {_MONTHS[day.month - 1]} {day.year}"


def _money(amount: Decimal) -> str:
    if amount == amount.to_integral_value():
        return f"${int(amount):,}"
    return f"${amount:,.2f}"


def clauses(policy: PolicySpec, v: VersionTerms) -> list[ClauseDef]:
    """The clause list of one version, in document order."""
    if v.effective_to is None:
        period = (
            f"This version of the policy applies to incidents that happen on or after "
            f"{_long_date(v.effective_from)}. It stays in force until we issue a later version "
            f"or the policy ends."
        )
    else:
        period = (
            f"This version of the policy applies to incidents that happen on or after "
            f"{_long_date(v.effective_from)} and before {_long_date(v.effective_to)}. An incident "
            f"on {_long_date(v.effective_to)} itself falls under the next version, if there is one."
        )

    out = [
        ClauseDef(
            "1.1",
            "Definitions",
            '"You" means the policyholder named in the schedule. "The vehicle" means the private '
            'car shown in the schedule. "Excess" means the amount you pay towards each claim '
            'before we pay anything. "Market value" means the cost of replacing the vehicle with '
            "one of the same make, model, age, mileage and condition immediately before the loss.",
        ),
        ClauseDef(
            "1.2",
            "Who may drive the vehicle",
            "Cover applies while the vehicle is driven by you or by a named driver listed in the "
            "schedule, provided the driver holds a driving licence that is valid on the date of "
            "the incident. A driver whose licence has expired, been suspended or been revoked is "
            "not covered, and neither is anyone else driving with your permission.",
        ),
        ClauseDef("2.1", "Period of cover", period),
        ClauseDef(
            "2.2",
            "Premium payment and lapse",
            "Premiums are payable monthly by direct debit. If a payment fails we will write to "
            "you. If it is still unpaid 14 days after that letter the policy lapses on the date "
            "given in our notice, and no cover applies to incidents on or after that date.",
        ),
        ClauseDef(
            "3.1",
            "Where you are covered",
            "Cover applies while the vehicle is in the United States or Canada. Trips into Mexico "
            "of up to 30 days are covered for collision only, provided you buy the local "
            "liability insurance that Mexican law requires.",
        ),
        ClauseDef(
            "4.1",
            "Collision damage",
            "We cover accidental damage to the vehicle caused by a collision with another "
            "vehicle, an object or an animal, including damage caused while the vehicle is "
            "parked. We pay the cost of repair or, if the vehicle is a total loss, its market "
            f"value, up to a limit of {_money(v.collision_limit)} per claim, less the collision "
            "excess in section 5.1.",
        ),
    ]
    if v.glass_limit is not None:
        out.append(
            ClauseDef(
                "4.2",
                "Glass breakage",
                "We cover repair or replacement of the windscreen, side windows, rear window and "
                "sunroof of the vehicle, including recalibration of cameras and sensors mounted "
                f"on the windscreen, up to a limit of {_money(v.glass_limit)} per claim. A glass "
                "claim on its own does not affect your no-claims discount. The glass excess in "
                "section 5.1 applies.",
            )
        )
    out.append(
        ClauseDef(
            "4.3",
            "Theft of the vehicle",
            "We cover loss of the vehicle by theft or attempted theft, and damage caused during "
            f"a theft or attempted theft, up to a limit of {_money(v.theft_limit)} per claim. "
            "The theft must be reported to the police within 48 hours of discovery and the "
            "police incident report sent with the claim. There is no cover if the keys were left "
            "in or on the vehicle.",
        )
    )
    if v.hire_car_daily is not None:
        out.append(
            ClauseDef(
                "4.4",
                "Replacement vehicle",
                "While the vehicle is off the road for repairs after a covered collision, or "
                "after it has been stolen, we pay for a hire car of similar size at up to "
                f"{_money(v.hire_car_daily)} a day for no more than 14 days.",
            )
        )

    excess_lines = [f"collision claims {_money(v.collision_excess)}"]
    if v.glass_excess is not None:
        excess_lines.append(f"glass claims {_money(v.glass_excess)}")
    excess_lines.append(f"theft claims {_money(v.theft_excess)}")
    out.append(
        ClauseDef(
            "5.1",
            "Excess",
            "You pay the following excess on each claim: "
            + "; ".join(excess_lines)
            + ". We deduct the excess from the amount we pay. If one incident leads to more "
            "than one type of claim, only the highest excess applies.",
        )
    )
    out += [
        ClauseDef(
            "6.1",
            "General exclusions",
            "We do not cover loss or damage that happens while the vehicle is driven by someone "
            "not allowed to drive it under section 1.2, while the driver is over the legal limit "
            "for alcohol or under the influence of drugs, while the vehicle is used for racing, "
            "rallies or speed tests, or while it is used for hire or reward, including "
            "ride-hailing. We do not cover deliberate damage caused by you or by anyone acting "
            "with your consent.",
        ),
        ClauseDef(
            "6.2",
            "Wear, tear and mechanical failure",
            "We do not cover wear and tear, loss of value, mechanical or electrical breakdown, or "
            "damage to tyres caused by braking, punctures, cuts or bursts.",
        ),
    ]
    if v.glass_limit is None:
        out.append(
            ClauseDef(
                "6.3",
                "Windscreen and window damage",
                "This version of the policy does not cover damage to the windscreen, windows or "
                "sunroof on its own. Glass broken in the same collision as other damage to the "
                "vehicle is paid as part of the collision claim under section 4.1.",
            )
        )
    if v.hire_car_daily is None:
        out.append(
            ClauseDef(
                "6.4",
                "Hire cars and replacement transport",
                "We do not pay for a hire car, taxi or other replacement transport while the "
                "vehicle is being repaired or after it has been stolen.",
            )
        )
    docs = (
        "Every claim must include a completed claim form and a copy of the claimant's driving "
        "licence. A collision claim also needs the repair invoice and a police or incident "
        "report."
    )
    if v.glass_limit is not None:
        docs += " A glass claim also needs the repair invoice."
    docs += " A theft claim also needs the police incident report."
    out += [
        ClauseDef(
            "7.1",
            "Making a claim",
            f"Tell us about any incident that may lead to a claim within {v.notice_days} days. "
            "Send the claim form signed by the person who was driving, together with the "
            "documents listed in section 7.2. We may ask for the vehicle to be inspected before "
            "repairs start.",
        ),
        ClauseDef("7.2", "Documents we need", docs),
        ClauseDef(
            "7.3",
            "Fraud and false documents",
            "If a claim is dishonest or supported by false documents, such as an altered invoice "
            "or a licence in another person's name, we will refuse the whole claim, may cancel "
            "the policy from the date of the dishonest act and may report the matter to the "
            "police.",
        ),
    ]
    return out


def render_markdown(policy: PolicySpec, v: VersionTerms) -> str:
    to = "open-ended" if v.effective_to is None else f"until {v.effective_to.isoformat()}"
    lines = [
        f"# {INSURER} private motor policy {policy.policy_number}, version {v.version_no}",
        "",
        f"In force from {v.effective_from.isoformat()}, {to}.",
        "",
    ]
    for clause in clauses(policy, v):
        lines += [f"## {clause.ref} {clause.title}", "", clause.text, ""]
    return "\n".join(lines)
