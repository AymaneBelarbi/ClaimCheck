"""The tools the adjuster agent may call. Nothing outside this set is ever registered.

Every tool reads, except the ones in WRITE_TOOLS. A write tool must refuse to act unless a
person has confirmed it, and the agent loop never lets the model give that confirmation.
"""

ALLOWED_TOOLS: frozenset[str] = frozenset(
    {
        "get_claim_summary",
        "list_claim_documents",
        "get_extracted_fields",
        "get_validation_results",
        "search_policy_clauses",
        "get_policy_version",
        "flag_for_review",
    }
)

WRITE_TOOLS: frozenset[str] = frozenset({"flag_for_review"})

assert WRITE_TOOLS <= ALLOWED_TOOLS
