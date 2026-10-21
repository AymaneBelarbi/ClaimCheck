"""Clause-retrieval questions: a policy, an incident date and the clause that answers it.

The answer is always taken from the version in force on that date. Some questions have a
different answering clause depending on the version (windscreen damage is an exclusion in v1 and
a cover clause from v2; hire cars likewise), which is what catches retrieval that ignores the
incident date.
"""

import random
from collections.abc import Callable
from datetime import date, timedelta

from evals.dataset.policies import LATEST_INCIDENT, POLICIES, VersionTerms, version_in_force
from evals.dataset.types import ClauseQuestion, DatasetError

Resolver = Callable[[VersionTerms], str]


def _fixed(ref: str) -> Resolver:
    return lambda _v: ref


# Asked against every version.
ALWAYS: tuple[tuple[str, Resolver], ...] = (
    (
        "Is a cracked windscreen covered?",
        lambda v: "4.2" if v.glass_limit is not None else "6.3",
    ),
    (
        "Will the insurer pay for a hire car while my car is being repaired?",
        lambda v: "4.4" if v.hire_car_daily is not None else "6.4",
    ),
    ("What is the most the policy pays for collision damage?", _fixed("4.1")),
    ("How much excess do I pay on a theft claim?", _fixed("5.1")),
)

# Three of these per version, rotated.
ROTATING: tuple[tuple[str, Resolver], ...] = (
    ("Am I covered if my car is stolen from my driveway?", _fixed("4.3")),
    ("How many days do I have to report an incident?", _fixed("7.1")),
    ("Which documents must I send with a collision claim?", _fixed("7.2")),
    ("Is a tyre puncture covered?", _fixed("6.2")),
    ("What happens if I miss a premium payment?", _fixed("2.2")),
    ("Am I covered if I drive the car into Canada?", _fixed("3.1")),
    ("Is the car covered if my partner drives it with my permission?", _fixed("1.2")),
    ("What happens if an invoice I send turns out to have been altered?", _fixed("7.3")),
    ("Is damage covered if the driver had been drinking?", _fixed("6.1")),
)


def _sample_day(rng: random.Random, v: VersionTerms) -> date:
    end = v.effective_to or (LATEST_INCIDENT + timedelta(days=1))
    span = (end - v.effective_from).days
    return v.effective_from + timedelta(days=rng.randrange(span))


def build_questions(rng: random.Random) -> list[ClauseQuestion]:
    rows: list[ClauseQuestion] = []
    rotation = 0
    for policy in POLICIES:
        for v in policy.versions:
            asked = list(ALWAYS)
            for _ in range(3):
                asked.append(ROTATING[rotation % len(ROTATING)])
                rotation += 1
            for i, (question, resolve) in enumerate(asked):
                # For later versions, ask one question on the first day of the version: the
                # boundary belongs to the new version, not the old one.
                day = v.effective_from if v.version_no > 1 and i == 0 else _sample_day(rng, v)
                if version_in_force(policy, day) is not v:
                    raise DatasetError(f"{policy.policy_number} {day}: not in v{v.version_no}")
                rows.append(
                    ClauseQuestion(
                        id=f"CQ-{len(rows) + 1:03d}",
                        policy_number=policy.policy_number,
                        incident_date=day,
                        question=question,
                        expected_clause_ref=resolve(v),
                        expected_version_no=v.version_no,
                    )
                )
    return rows
