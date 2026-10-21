import type { ReviewFile } from "./types";

// A claim with one passing and two blocking rules, shaped like claimcheck.core.review.ReviewFile.
export const reviewFixture: ReviewFile = {
  claim: {
    id: "11111111-1111-1111-1111-111111111111",
    packet_id: "PKT-007",
    status: "needs_review",
    claim_type: "collision",
    incident_date: "2026-03-14",
  },
  policy_version: { version_no: 2, effective_from: "2026-01-01", effective_to: null },
  documents: [
    {
      id: "d1",
      filename: "invoice.pdf",
      doc_type: "invoice",
      classification_confidence: 0.91,
      classification_method: "embedding",
      ocr_mean_confidence: 0.88,
      low_confidence: false,
      error: null,
    },
  ],
  fields: [
    {
      document_id: "d1",
      doc_type: "invoice",
      field_name: "total_amount",
      value: "1840.00",
      confidence: 0.64,
      verdict: "low_confidence",
      evidence: { document_id: "d1", page: 1, span: [120, 128], text: "Total 1840.00" },
    },
  ],
  rules: [
    { rule_id: "policy_in_force", description: "Policy covers the incident date", verdict: "found", message: "ok", evidence: [] },
    {
      rule_id: "name_match",
      description: "Holder name matches across documents",
      verdict: "conflicting",
      message: "claim form and id card names differ",
      evidence: [{ document_id: "d1", page: 1, span: [4, 20], text: "A. Martin" }],
    },
    { rule_id: "required_docs", description: "All required documents present", verdict: "missing", message: "no incident report", evidence: [] },
  ],
  summary: {
    verdict_counts: { found: 1, missing: 1, conflicting: 1, low_confidence: 0 },
    field_verdict_counts: { found: 0, missing: 0, conflicting: 0, low_confidence: 1 },
    blocking: ["name_match", "required_docs"],
    ready_for_decision: false,
  },
};
