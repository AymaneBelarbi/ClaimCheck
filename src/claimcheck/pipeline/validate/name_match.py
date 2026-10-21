import unicodedata

from rapidfuzz import fuzz

from claimcheck.core.types import ClaimType, DocType
from claimcheck.pipeline.validate.base import ClaimContext, RuleResult, read_inputs

NAME_MATCH_THRESHOLD = 90.0


def normalize_name(name: str) -> str:
    """Casefolded with accents removed, so "José Álvarez" and "JOSE ALVAREZ" compare equal."""
    decomposed = unicodedata.normalize("NFKD", name)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def names_agree(a: str, b: str) -> bool:
    """Whether two spellings are the same name by R1's own test (normalized, order-free, >= 90)."""
    return fuzz.token_sort_ratio(normalize_name(a), normalize_name(b)) >= NAME_MATCH_THRESHOLD


class NameMatch:
    rule_id = "R1_NAME_MATCH"
    description = "The name on the ID card matches the claimant on the claim form."
    applies_to = frozenset(ClaimType)

    def evaluate(self, ctx: ClaimContext) -> RuleResult:
        holder_spec = (DocType.ID_CARD, "full_name")
        claimant_spec = (DocType.CLAIM_FORM, "claimant_name")
        # Two ID cards (or two forms) spelling one name differently still agree with each other.
        inputs = read_inputs(
            ctx,
            holder_spec,
            claimant_spec,
            agree={holder_spec: names_agree, claimant_spec: names_agree},
        )
        if (blocked := inputs.blocked(self.rule_id)) is not None:
            return blocked
        holder = inputs.get(DocType.ID_CARD, "full_name")
        claimant = inputs.get(DocType.CLAIM_FORM, "claimant_name")
        # Floored so the message never shows "90 < 90" for a score of 89.6.
        score = int(fuzz.token_sort_ratio(normalize_name(holder), normalize_name(claimant)))
        if score >= NAME_MATCH_THRESHOLD:
            return inputs.result(
                self.rule_id,
                True,
                f"Name on ID '{holder}' matches claimant '{claimant}' (score {score})",
            )
        return inputs.result(
            self.rule_id,
            False,
            f"Name on ID '{holder}' does not match claimant '{claimant}' "
            f"(score {score} < {NAME_MATCH_THRESHOLD:g})",
        )
