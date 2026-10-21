# ADR-002: Policy versioning

Status: accepted, Oct 20, 2026

## Context

A policy changes over its life: limits move, exclusions are added, wording is tightened. A claim has to be judged against the wording that applied on the day of the incident, not the wording that is current when an adjuster opens the file. A glass claim from March must not be refused because an exclusion was added in September.

This matters most for the retrieval step. If clauses from every version sit in one index and the search simply returns the closest match, the nearest clause may well come from the wrong year, and the answer would cite it with full confidence.

## Decision

**Versions are immutable and never deleted.** A policy is one row in `policies`. Each revision is a row in `policy_versions` with a `version_no`, an `effective_from` date, and an `effective_to` date. Changing a policy means closing the current version (setting `effective_to`) and inserting the next one. A past version is never edited, because claims decided under it must stay explainable. Clauses belong to a version (`policy_clauses.policy_version_id`), so each version carries its own complete clause set.

**Date ranges are half-open.** A version covers `effective_from <= d < effective_to`: the start day is included and the end day is not. `effective_to NULL` means the version is current and has no end. With this convention a version that ends on 2025-07-01 and its successor that starts on 2025-07-01 meet exactly, with no gap and no shared day, and the successor's `effective_from` is simply the predecessor's `effective_to`. There is no "last day" to argue about. A check constraint requires `effective_to > effective_from`.

**The database enforces non-overlap.** An exclusion constraint on `policy_versions` rejects two rows for the same policy whose `daterange(effective_from, effective_to, '[)')` overlap (GiST index, `policy_id WITH =`, range `WITH &&`). An open-ended version has an unbounded range, so it also conflicts with any later version, which is the intended result: the current version must be closed before the next one is inserted. This makes "at most one version in force on any day" a property of the data rather than of application code.

**One function answers "which version applied".** `db.policy_in_force.policy_in_force(session, policy_id, incident_date)` returns the matching `PolicyVersion` or `None`. The filter is exactly `effective_from <= :d AND :d < coalesce(effective_to, 'infinity'::date)`. It returns `None` for an unknown policy, a date before the first version, or a gap between versions; callers treat that as "no coverage found" and surface it, they do not fall back to the nearest version. If more than one row matches it raises `OverlappingVersionsError`. The constraint should make that impossible, but a restored backup or a hand-edited table could break it, and an arbitrary pick between two versions is a worse failure than a loud one.

**Every clause lookup goes through this function.** Retrieval filters `policy_clauses` by the version id that `policy_in_force` returns, in SQL and before any similarity ranking. The vector search never sees clauses from other versions, so it cannot rank one above the right one. The coverage rule in validation, the retrieval eval and the agent's clause tool all use the same path. Nothing queries clauses by policy alone.

## Alternatives considered

- **Mutate the policy in place and keep the history in an audit table.** Simpler schema, but every read of an old claim then needs a point-in-time reconstruction, and clauses would need the same treatment. Versions as rows make the as-of query one indexed filter.
- **Closed ranges (`effective_to` is the last covered day).** Reads naturally, but adjacent versions need `to = from - 1 day`, which invites off-by-one bugs and leaves a gap for any timestamp-valued date. Half-open ranges compose without arithmetic.
- **Rely on application code to keep versions from overlapping.** Cheaper to write, but a second writer, a migration script or a manual fix can break it, and the symptom (two plausible versions) appears far from the cause. The constraint is a few lines and cannot be bypassed.
- **Filter by version after retrieval.** Retrieve the top k clauses across all versions and drop the wrong ones. Rejected: with several versions of near-identical wording, the top k can be filled by the wrong versions and leave nothing correct to return. The filter has to come first.

## Consequences

- Storage grows with each version, since a new version carries its own clauses even where the text is unchanged. For one insurer with a handful of versions per policy this is negligible, and it keeps every lookup a filter on one version id, with no inheritance rules between versions.
- Publishing a new version is two steps in one transaction: close the current one, insert the next. Doing it in the other order fails on the constraint, which is the intended safeguard.
- A claim whose incident date predates a policy's first version, or falls in a gap, finds no version. The validation rule reports that as a verdict with evidence instead of guessing.
- `incident_date` is a calendar date. Time-of-day and time zones are out of scope. The date declared when the claim is filed selects the version, so rule R7 (`pipeline/validate/incident_date_consistent.py`) checks it against the claim form and, when there is one, the incident report; a disagreement blocks the decision with both documents as evidence.
- The function is tested against the boundaries (start inclusive, end exclusive, open-ended, before the first version, unknown policy) and against a table with the constraint removed, to cover the defensive check.
