"""Split a policy version's markdown into clauses.

A clause is one `## <ref> <title>` section. Coverage type and limit are read out of the text
so that the validation rules and retrieval filters do not each parse the policy again.
"""

import re
from dataclasses import dataclass
from decimal import Decimal

_HEADING = re.compile(r"^## (?P<ref>\d+(?:\.\d+)*) (?P<title>.+?)\s*$", re.MULTILINE)
_LIMIT = re.compile(r"limit of \$([\d,]+(?:\.\d{2})?)")

# Matched against the lower-cased title. Titles of exclusions that merely mention glass
# ("Windscreen and window damage") carry no limit and are deliberately left untyped.
_COVERAGE_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("collision", "collision"),
    ("glass", "glass"),
    ("theft", "theft"),
)


@dataclass(frozen=True)
class Clause:
    clause_ref: str
    title: str
    text: str
    coverage_type: str | None
    limit_amount: Decimal | None


def chunk_clauses(markdown: str) -> list[Clause]:
    headings = list(_HEADING.finditer(markdown))
    clauses = []
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(markdown)
        text = markdown[heading.end() : end].strip()
        title = heading.group("title")
        clauses.append(
            Clause(
                clause_ref=heading.group("ref"),
                title=title,
                text=text,
                coverage_type=_coverage_type(title),
                limit_amount=_limit(text),
            )
        )
    return clauses


def _coverage_type(title: str) -> str | None:
    lowered = title.casefold()
    for keyword, coverage_type in _COVERAGE_KEYWORDS:
        if keyword in lowered:
            return coverage_type
    return None


def _limit(text: str) -> Decimal | None:
    match = _LIMIT.search(text)
    return Decimal(match.group(1).replace(",", "")) if match else None
